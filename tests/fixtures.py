"""Dependency-free, deterministic checkpoints; not representative language models."""
import array
import math
import struct
from pathlib import Path

CONFIG = (4, 8, 2, 2, 1, 128010, 32)


def floats(n, phase):
    return array.array('f', (0.12 * math.sin(i * 0.71 + phase) for i in range(n)))


def create(root, shared=True, zero=False, stop=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    d, h, layers, heads, kvheads, v, seq = CONFIG
    kv = d // heads * kvheads
    emb = floats(v * d, 0.2)
    norms = array.array('f', [1] * (layers * d))
    matrices = [(d*d, 1), (d*kv, 2), (d*kv, 3), (d*d, 4), (d*h, 5), (h*d, 6), (d*h, 7)]
    weights = [emb] + [floats(n, phase + layer) for n, phase in matrices for layer in range(layers)]
    if not shared:
        classifier = array.array('f', [0] * (v*d))
        classifier[65*d:66*d] = array.array('f', [10, -10, 10, -10])
        weights.append(classifier)
    if stop:
        if shared:
            raise ValueError('stop fixtures require separate classifier')
        emb[:] = array.array('f', [1,0,0,0]) * v
        if stop == 'after_one':
            emb[65*d] = -1
        weights = [emb] + [array.array('f', [0]) * len(w) for w in weights[1:]]
        weights[-1][128001*d] = 100 if stop == 'first' else -100
        if stop == 'after_one':
            weights[-1][65*d] = 100
    if zero:
        weights = [array.array('f', [0]) * len(w) for w in weights]
    with (root/'fp32.bin').open('wb') as f:
        f.write(struct.pack('<7i', d, h, layers, heads, kvheads, v if shared else -v, seq))
        f.write(emb.tobytes() if not zero else bytes(len(emb)*4))
        f.write(norms.tobytes())
        for w in weights[1:1+4*layers]:
            f.write(w.tobytes())
        f.write(norms.tobytes())
        for w in weights[1+4*layers:1+7*layers]:
            f.write(w.tobytes())
        f.write(array.array('f', [1]*d).tobytes())
        f.write(bytes(seq*(d//heads)*4))
        if not shared:
            f.write(weights[-1].tobytes())
    gs = 2
    with (root/'int8.bin').open('wb') as f:
        header = struct.pack('<Ii7iBi', 0x616b3432, 2, *CONFIG, int(shared), gs)
        f.write(header + bytes(256-len(header)))
        f.write(norms.tobytes()*2 + array.array('f', [1]*d).tobytes())
        for w in weights:
            q, scales = array.array('b'), array.array('f')
            for i in range(0, len(w), gs):
                block = w[i:i+gs]
                scale = max(map(abs, block))/127
                scales.append(scale)
                q.extend(round(x/scale) if scale else 0 for x in block)
            f.write(q.tobytes())
            f.write(scales.tobytes())
    with (root/'tokenizer.bin').open('wb') as f:
        f.write(struct.pack('<I', 24))
        for i in range(v):
            piece = bytes([i-3]) if 3 <= i < 259 else f'<token{i}>'.encode()
            f.write(struct.pack('<fi', -i, len(piece)) + piece)
    return root


if __name__ == '__main__':
    import sys
    create(sys.argv[1])
