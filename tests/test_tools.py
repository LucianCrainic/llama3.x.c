"""Converter smoke checks without installing large ML dependencies."""
import ast
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import struct
from typing import List
import unittest

TOOLS = Path(__file__).resolve().parents[1]/'tools'


class ToolSmokeTests(unittest.TestCase):
    def test_engine_comparison_gate(self):
        spec = importlib.util.spec_from_file_location('compare_engines', TOOLS/'compare_engines.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        def compare(left, right):
            return module.compare(io.BytesIO(struct.pack('4f', *left)),
                                  io.BytesIO(struct.pack('4f', *right)), 2)
        values = [1, 2, 4, 3]
        self.assertTrue(compare(values, values)['passed'])
        different = compare(values, [1, 2, 4, 5])
        self.assertFalse(different['passed'])
        self.assertEqual(different['max_abs'], 2)
        self.assertEqual(different['argmax_agreement'], .5)
        self.assertFalse(compare([0, 2, 4, 3], [-0., 2, 4, 3])['bitwise'])
        self.assertFalse(compare([float('nan')]*4, [float('nan')]*4)['passed'])
        with self.assertRaises(ValueError):
            module.compare(io.BytesIO(b''), io.BytesIO(b''), 2)

    def test_all_sources_compile(self):
        for path in TOOLS.glob('*.py'):
            compile(path.read_text(), str(path), 'exec')

    def test_tokenizer_encode_delegates_and_adds_special_tokens(self):
        # Isolate the real method so the test needs no tiktoken installation or
        # licensed tokenizer.model. This checks the wrapper, not BPE correctness.
        source = ast.parse((TOOLS/'tokenizer.py').read_text())
        cls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'Tokenizer')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encode')
        module = ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
        namespace = {'List':List}
        exec(compile(module,'tokenizer.py','exec'),namespace)
        calls = []
        def model_encode(text, **kwargs):
            calls.append((text, kwargs))
            return [7,8]
        instance = SimpleNamespace(model=SimpleNamespace(encode=model_encode),bos_id=128000,eos_id=128001)
        result = namespace['encode'](instance,'hello è',True,True,{'allowed'},set())
        self.assertEqual(result,[128000,7,8,128001])
        self.assertEqual(calls,[('hello è',{'allowed_special':{'allowed'},'disallowed_special':set()})])


if __name__ == '__main__':
    unittest.main()
