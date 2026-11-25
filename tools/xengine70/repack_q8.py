#!/usr/bin/env python3
"""Repack llama.cpp Llama 3.1 70B Q8_0 into the C format without requantizing."""
import argparse
from pathlib import Path
import struct

from gguf import GGUFReader, GGMLQuantizationType
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("reference", type=Path, help="v3 C checkpoint for the verified configuration header")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.reference.open("rb") as source:
        header = bytearray(source.read(256))
    fields = struct.unpack_from("<12I", header)
    if fields != (0x616B3432, 3, 8192, 28672, 80, 64, 8, 128256, 131072, 1, 0, 64):
        raise ValueError(f"Unexpected Llama 3.1 70B Q8 header: {fields}")
    struct.pack_into("<I", header, 44, 32)
    tensors = {t.name: t for t in GGUFReader(args.gguf).tensors}
    blocks = np.dtype([("scale", "<f2"), ("q", "i1", (32,))])
    layers = range(80)
    d, hidden, kv, vocab = 8192, 28672, 1024, 128256
    norms = [f"blk.{i}.{role}.weight" for role in ("attn_norm", "ffn_norm") for i in layers]
    norms.append("output_norm.weight")
    roles = [("attn_q", d, d), ("attn_k", kv, d), ("attn_v", kv, d), ("attn_output", d, d),
             ("ffn_gate", hidden, d), ("ffn_down", d, hidden), ("ffn_up", hidden, d)]
    weights = [("token_embd.weight", vocab, d)]
    weights.extend((f"blk.{i}.{role}.weight", rows, columns) for role, rows, columns in roles for i in layers)
    weights.append(("output.weight", vocab, d))

    with args.output.open("xb") as out:
        out.write(header)
        for name in norms:
            tensor = tensors[name]
            if tensor.tensor_type != GGMLQuantizationType.F32 or tuple(tensor.shape) != (d,):
                raise ValueError(f"Unexpected norm representation: {name}")
            np.asarray(tensor.data, dtype="<f4").tofile(out)
        for index, (name, rows, columns) in enumerate(weights):
            tensor = tensors[name]
            if tensor.tensor_type != GGMLQuantizationType.Q8_0 or tuple(tensor.shape) != (columns, rows):
                raise ValueError(f"Unexpected weight representation: {name}")
            records = tensor.data.reshape(-1).view(blocks)
            values = np.ascontiguousarray(records["q"])
            if values.min() < -127:
                raise ValueError(f"Unsupported -128 value in {name}")
            values.tofile(out)
            records["scale"].astype("<f4").tofile(out)
            if index % 80 == 0:
                print(name, out.tell(), flush=True)


if __name__ == "__main__":
    main()
