"""Converter smoke checks without installing large ML dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import List
import unittest

TOOLS = Path(__file__).resolve().parents[1]/'tools'


class ToolSmokeTests(unittest.TestCase):
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
