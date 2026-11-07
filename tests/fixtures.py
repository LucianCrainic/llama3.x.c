"""Dependency-free, deterministic checkpoints; not representative language models."""
import array
import math
import struct
from pathlib import Path

CONFIG = (4, 8, 2, 2, 1, 128010, 32)
# B: two KV heads and an even/odd vocabulary, so the model can be split over two sockets; Q8 group size 4 keeps the
# per-socket column slices (8 and 16) aligned to groups.
TP_CONFIG = (16, 32, 2, 4, 2, 128010, 64)
TP_ODD_CONFIG = (16, 32, 2, 4, 2, 128011, 64)


def floats(n, phase):
    return array.array('f', (0.12 * math.sin(i * 0.71 + phase) for i in range(n)))


def create(root, shared=True, zero=False, stop=None, seq_len=None, config=None, gs=2):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    d, h, layers, heads, kvheads, v, seq = config or CONFIG
    if seq_len is not None:
        seq = seq_len
    kv = d // heads * kvheads
    emb = floats(v * d, 0.2)
    norms = array.array('f', [1] * (layers * d))
    matrices = [(d*d, 1), (d*kv, 2), (d*kv, 3), (d*d, 4), (d*h, 5), (h*d, 6), (d*h, 7)]
    weights = [emb] + [floats(n, phase + layer) for n, phase in matrices for layer in range(layers)]
    if not shared:
        classifier = array.array('f', [0] * (v*d))
        classifier[65*d:66*d] = array.array('f', [10, -10] * (d // 2))
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
    with (root/'int8.bin').open('wb') as f:
        header = struct.pack('<Ii7iBi', 0x616b3432, 2, d, h, layers, heads, kvheads, v, seq, int(shared), gs)
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


def upgrade_v3(root, *, epsilon=1e-5, theta=500000., scaling=False):
    """Upgrade an already generated legacy fixture without ML dependencies."""
    root = Path(root)
    d, h, layers, heads, kvheads, vocab, seq = CONFIG
    fp = (root/'fp32.bin').read_bytes()
    shared = struct.unpack_from('<i', fp, 20)[0] > 0
    def header(precision, gs):
        data = bytearray(256)
        struct.pack_into('<9I', data, 0, 0x616b3432, 3, *CONFIG)
        struct.pack_into('<3I2fI3f3I4i', data, 36, precision, shared, gs, epsilon, theta,
                         int(scaling), 32., 1., 4., 8192, seq, d//heads, 128000, 128001, 128009, 128008)
        data[100:116] = b'synthetic-phase2'
        return data
    # The v0 unused frequency arrays precede an optional classifier.
    before_tables = len(fp) - seq*(d//heads)*4 - (0 if shared else vocab*d*4)
    payload = fp[28:before_tables] + fp[before_tables+seq*(d//heads)*4:]
    (root/'fp32.bin').write_bytes(header(0, 0) + payload)
    quant = (root/'int8.bin').read_bytes()
    (root/'int8.bin').write_bytes(header(1, 2) + quant[256:])
    original = (root/'tokenizer.bin').read_bytes()
    data = bytearray(struct.pack('<4I7iI', 0x336b6f74, 2, vocab, 128000, 24,
                                128000, 128001, 128009, 128008, 128006, 128007, 0))
    offset = 4
    for rank in range(vocab):
        _, length = struct.unpack_from('<fi', original, offset)
        offset += 8
        data.extend(struct.pack('<fI', rank, length))
        data.extend(original[offset:offset+length])
        offset += length
    (root/'tokenizer.bin').write_bytes(data)
    return root


if __name__ == '__main__':
    import sys
    create(sys.argv[1])
