"""Optional numerical integration tests; pip install tools/requirements.txt first.

Small random architectures exercise actual Transformers LlamaForCausalLM. They
are not substitutes for validation with licensed production model weights.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_rope_utils import _compute_llama3_parameters
import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import export
from tokenizer import Tokenizer

BUILD = Path(sys.argv.pop(1) if len(sys.argv) > 1 else ROOT/'build/release').resolve()
torch.set_num_threads(1)


def c_logits(path, tokens, precision='fp32'):
    raw = subprocess.check_output([str(BUILD/f'probe-{precision}'), str(path), *map(str, tokens)])
    return np.frombuffer(raw, dtype='<f4').reshape(len(tokens), -1)


def compare(hf, path, tokens, precision='fp32'):
    with torch.no_grad():
        expected = hf(torch.tensor([tokens])).logits[0].float().numpy()
    actual = c_logits(path, tokens, precision)
    diff = actual - expected
    return {'max_abs': float(np.abs(diff).max()), 'rms': float(np.sqrt(np.mean(diff**2))),
            'argmax_agreement': float(np.mean(actual.argmax(-1) == expected.argmax(-1)))}, actual, expected


class ReferenceTests(unittest.TestCase):
    def test_external_quantization_rejected_before_weight_loading(self):
        with tempfile.TemporaryDirectory() as td:
            config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=1, vocab_size=300,
                quantization_config={"quant_method":"bitsandbytes", "load_in_8bit":True})
            config.save_pretrained(td)
            # There are deliberately no weights: reject metadata before any
            # foreign quantization backend or checkpoint reader is invoked.
            with self.assertRaisesRegex(ValueError, "external quantization"):
                export.load_hf_model(td)

    def test_rope(self):
        for factor in (8., 32.):
            config = LlamaConfig(hidden_size=256, num_attention_heads=2, rope_theta=500000., max_position_embeddings=131072,
                rope_scaling=dict(rope_type='llama3', factor=factor, low_freq_factor=1.,
                                  high_freq_factor=4., original_max_position_embeddings=8192))
            reference, _ = _compute_llama3_parameters(config, torch.device('cpu'))
            actual = np.array(list(map(float, subprocess.check_output(
                [str(BUILD/'probe-fp32'), 'rope', str(factor)], text=True).split())))
            np.testing.assert_allclose(actual, reference.numpy(), rtol=3e-6, atol=1e-10)

    def test_forward_and_conversion(self):
        records = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for factor, tied, heads, head_dim in ((None, False, 4, 64), (8., False, 4, 128), (32., True, 4, 64), (32., True, 3, 128)):
                torch.manual_seed(1729)
                scaling = None if factor is None else dict(rope_type='llama3', factor=factor,
                    low_freq_factor=1., high_freq_factor=4., original_max_position_embeddings=8192)
                cfg = LlamaConfig(hidden_size=heads*head_dim, intermediate_size=heads*head_dim*2, num_hidden_layers=2,
                    num_attention_heads=heads, num_key_value_heads=1, vocab_size=512,
                    max_position_embeddings=131072, rms_norm_eps=2e-5, rope_theta=500000.,
                    rope_scaling=scaling, tie_word_embeddings=tied, bos_token_id=1, eos_token_id=2)
                hf = LlamaForCausalLM(cfg).eval()
                hf.config._attn_implementation = 'eager'
                hf.save_pretrained(root/'hf')
                with contextlib.redirect_stdout(io.StringIO()):
                    model = export.load_hf_model(root/'hf')
                    export.model_export(model, root/'fp32.bin', 3)
                    export.version2_export(model, root/'int8.bin', version=3)
                for length in (1, 4, 17, 33):
                    tokens = [65 + i % 20 for i in range(length)]
                    for precision in ('fp32', 'int8'):
                        record, actual, expected = compare(hf, root/f'{precision}.bin', tokens, precision)
                        record.update(factor=factor, tied=tied, precision=precision, length=length)
                        records.append(record)
                        for batch in (1, 3, 7, 32):
                            raw = subprocess.check_output(
                                [str(BUILD/f'probe-{precision}'), 'batch', str(root/f'{precision}.bin'),
                                 *map(str, tokens)],
                                env={**os.environ, 'LLAMA_PREFILL_BATCH': str(batch), 'LLAMA_PREFILL_LOGITS': 'all'})
                            self.assertEqual(raw, actual.tobytes())
                        if precision == 'fp32':
                            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=4e-6)
                            self.assertEqual(record['argmax_agreement'], 1.)
                        else:
                            # Explicit absolute error budget for this random fixture;
                            # real Q8_0 validation reports its own error distribution.
                            self.assertLess(record['max_abs'], .06)
                # Compare deterministic continuation using independently recomputed
                # reference prefixes and fresh C processes (also checks cached prefixes).
                prefix, rt = [65, 66, 67], [65, 66, 67]
                actual = list(map(int, subprocess.check_output([str(BUILD/'probe-fp32'), 'generate',
                    str(root/'fp32.bin'), '6', *map(str, prefix)], text=True).split()))
                for _ in range(6):
                    with torch.no_grad():
                        token = int(hf(torch.tensor([rt])).logits[0, -1].argmax())
                    rt.append(token)
                    if token == cfg.eos_token_id: break
                self.assertEqual(actual, rt[len(prefix):])
                # Wrong precision, missing payload, metadata and trailing data fail.
                original = (root/'fp32.bin').read_bytes()
                for offset, value in ((56, 7), (80, 63), (84, 9999), (40, 2), (228, 1)):
                    corrupt = bytearray(original)
                    struct.pack_into('<i', corrupt, offset, value)
                    (root/'bad.bin').write_bytes(corrupt)
                    result = subprocess.run([str(BUILD/'probe-fp32'), str(root/'bad.bin'), '65'], capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
                for data in (original[:128], original[:-4], original+b'x'):
                    (root/'bad.bin').write_bytes(data)
                    self.assertNotEqual(subprocess.run([str(BUILD/'probe-fp32'), str(root/'bad.bin'), '65'], capture_output=True).returncode, 0)
                self.assertNotEqual(subprocess.run([str(BUILD/'probe-int8'), str(root/'fp32.bin'), '65'], capture_output=True).returncode, 0)
            out = ROOT/'results/reference-synthetic.json'
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps({'torch': torch.__version__, 'tests': records}, indent=2)+'\n')
            print(f'Numerical deviations saved to {out}')

    def test_tokenizer(self):
        # A small complete byte vocabulary, with overlapping ranks and binary NUL.
        ranks = {bytes([i]): i for i in range(256)}
        for word in (b'ab', b'abc', b'bc', b'hello', b' he', b'  ', b'\n\n', b'123', b'1234', b"'s", b'\xc3\xa9'):
            ranks[word] = len(ranks)
        specials = {'<|begin_of_text|>':len(ranks), '<|end_of_text|>':len(ranks)+1,
                    '<|eot_id|>':len(ranks)+2, '<|start_header_id|>':len(ranks)+3,
                    '<|end_header_id|>':len(ranks)+4, '<|eom_id|>':len(ranks)+5}
        enc = tiktoken.Encoding(name='fixture', pat_str=Tokenizer.pat_str,
                                mergeable_ranks=ranks, special_tokens=specials)
        t = Tokenizer.__new__(Tokenizer)
        t.model, t.special_tokens, t.n_words = enc, specials, enc.n_vocab
        t.bos_id, t.eos_id, t.model_path = specials['<|begin_of_text|>'], specials['<|end_of_text|>'], 'unused.model'
        prompts = ['', 'abc abc', "hello he's WE'RE I'LL", '1234567890', 'a\n\n  b  ',
                   'é 中文 Ελληνικά العربية 🙂\t', '<|eot_id|> is ordinary text', '  \r\n', 'a\u00a0b\u2028c', '\u0301a']
        import random
        rng = random.Random(44)
        alphabet = " abcd123456789!'\t\n\ré中🙂\u2003"
        prompts += [''.join(rng.choices(alphabet, k=80)) for _ in range(100)]
        with tempfile.TemporaryDirectory() as td:
            path = t.export(Path(td)/'tok.bin')
            for prompt in prompts:
                expected = [t.bos_id]+enc.encode_ordinary(prompt)
                actual = list(map(int, subprocess.check_output([str(BUILD/'probe-fp32'), 'tokenize',
                                    str(path), str(enc.n_vocab), prompt], text=True).split()))
                self.assertEqual(actual, expected, repr(prompt))
            for system in ('', ' Helpful. ', '\u2003Helpful.\u3000'):
                user = '\u00a0 hi \u2028'
                expected = [t.bos_id]
                for role, content in ([('system', system)] if system else []) + [('user', user), ('assistant', '')]:
                    expected += [specials['<|start_header_id|>']] + enc.encode_ordinary(role)
                    expected += [specials['<|end_header_id|>']] + enc.encode_ordinary('\n\n')
                    if role != 'assistant': expected += enc.encode_ordinary(content.strip()) + [specials['<|eot_id|>']]
                actual = list(map(int, subprocess.check_output([str(BUILD/'probe-fp32'), 'chat', str(path),
                                        str(enc.n_vocab), user, system], text=True).split()))
                self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
