# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 3 Community License Agreement.

import argparse
import os
import json
import struct
from pathlib import Path
from typing import List

import tiktoken
from tiktoken.load import load_tiktoken_bpe

TOKENIZER_MODEL = "tokenizer.model"  # the llama tiktoken tokenizer model


class Tokenizer:
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        mergeable_ranks = load_tiktoken_bpe(model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        num_base_tokens = len(mergeable_ranks)
        num_reserved_special_tokens = 256

        special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        self.n_words = self.model.n_vocab
        self.bos_id = self.special_tokens["<|begin_of_text|>"]
        self.eos_id = self.special_tokens["<|end_of_text|>"]
        self.pad_id = -1
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }

    def encode(
        self, s: str, bos: bool, eos: bool, allowed_special, disallowed_special
    ) -> List[int]:
        assert type(s) is str
        t = self.model.encode(
            s,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )

        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: List[int]) -> str:
        return self.model.decode(t)

    @classmethod
    def from_hf(cls, directory, chat_date="26 Jul 2024"):
        """Use tokenizer.json IDs directly, reversing the GPT-2 byte alphabet.

        Special spellings are read from added_tokens, never guessed from version.
        Unsupported tokenizer models/pretokenizers fail rather than approximating.
        """
        obj = cls.__new__(cls)
        path = Path(directory) / "tokenizer.json"
        data = json.loads(path.read_text())
        if data["model"]["type"] != "BPE" or data.get("normalizer"):
            raise ValueError("requires an unnormalized Llama byte-level BPE tokenizer")
        pre = data.get("pre_tokenizer", {})
        pieces = pre.get("pretokenizers", []) if pre.get("type") == "Sequence" else []
        if (len(pieces) != 2 or pieces[0].get("type") != "Split" or
            pieces[0].get("behavior") != "Isolated" or pieces[0].get("invert", False) or
            pieces[1].get("type") != "ByteLevel" or pieces[1].get("add_prefix_space", False) or
            pieces[1].get("use_regex", True)):
            raise ValueError("requires isolated regex split followed by ByteLevel without prefix space/regex")
        if data["model"].get("dropout") or data["model"].get("byte_fallback", False):
            raise ValueError("unsupported BPE dropout/byte fallback")
        # The exact regex is a compatibility boundary of the portable C runtime.
        def patterns(node):
            if isinstance(node, dict):
                if node.get("type") == "Split":
                    yield node.get("pattern", {}).get("Regex")
                for v in node.values():
                    yield from patterns(v)
            elif isinstance(node, list):
                for v in node:
                    yield from patterns(v)
        if list(patterns(data.get("pre_tokenizer"))) != [cls.pat_str]:
            raise ValueError("unsupported pretokenizer; requires the Llama 3 regex and ByteLevel")
        # ByteLevel maps the 256 bytes bijectively onto printable Unicode chars.
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(161, 173)) + list(range(174, 256))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + n); n += 1
        reverse = dict(zip(map(chr, cs), bs))
        ranks = {bytes(reverse[c] for c in token): rank for token, rank in data["model"]["vocab"].items()}
        if sorted(ranks.values()) != list(range(len(ranks))):
            raise ValueError("base ranks must be contiguous")
        obj.special_tokens = {t["content"]: t["id"] for t in data["added_tokens"] if t["special"]}
        obj.model = tiktoken.Encoding(name=Path(directory).name, pat_str=cls.pat_str,
                                     mergeable_ranks=ranks, special_tokens=obj.special_tokens)
        # Render a basic system/user exchange to extract the model-specific
        # prefix (for example the 3.1/3.2 knowledge/date header). Refuse templates
        # that cannot be represented by the runtime's simple message grammar.
        from transformers import AutoTokenizer
        hf = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        obj.chat_prefix = ""
        if hf.chat_template:
            marker, user = "LLAMA_SYSTEM_SENTINEL", "LLAMA_USER_SENTINEL"
            rendered = hf.apply_chat_template([{"role":"system", "content":marker},
                {"role":"user", "content":user}], tokenize=False, add_generation_prompt=True, date_string=chat_date)
            start = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            tail = (marker + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
            if not rendered.startswith(start) or not rendered.endswith(tail):
                raise ValueError("unsupported basic chat template")
            obj.chat_prefix = rendered[len(start):-len(tail)]
        obj.model_path = str(path)
        obj.n_words = obj.model.n_vocab
        obj.bos_id = obj.special_tokens["<|begin_of_text|>"]
        obj.eos_id = obj.special_tokens["<|end_of_text|>"]
        obj.stop_tokens = {obj.eos_id, obj.special_tokens["<|eot_id|>"]}
        return obj

    def export(self, output=None, legacy=False):
        tokens = [self.model.decode_single_token_bytes(i) for i in range(self.n_words)]
        max_length = max(map(len, tokens))
        output = Path(output) if output else Path(self.model_path).with_suffix(".bin")
        if output.resolve() == Path(self.model_path).resolve():
            raise ValueError("output would overwrite tokenizer source")
        with output.open("wb") as f:
            if legacy:
                f.write(struct.pack("<I", max_length))
            else:
                def special(name):
                    return self.special_tokens.get(name, -1)
                prefix = getattr(self, "chat_prefix", "").encode("utf-8")
                f.write(struct.pack("<4I7iI", 0x336b6f74, 2, self.n_words, len(self.model._mergeable_ranks),
                                    max_length, self.bos_id, self.eos_id, special("<|eot_id|>"),
                                    special("<|eom_id|>"), special("<|start_header_id|>"),
                                    special("<|end_header_id|>"), len(prefix)))
                f.write(prefix)
            for rank, piece in enumerate(tokens):
                f.write(struct.pack("<fI", rank, len(piece)))
                f.write(piece)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-t", "--tokenizer-model")
    source.add_argument("--hf", help="local Hugging Face model directory")
    parser.add_argument("-o", "--output")
    parser.add_argument("--chat-date", default="26 Jul 2024", help="fixed date rendered into instruct system prefix")
    parser.add_argument("--legacy", action="store_true", help="unverified original C tokenizer format")
    args = parser.parse_args()
    t = Tokenizer.from_hf(args.hf, args.chat_date) if args.hf else Tokenizer(args.tokenizer_model)
    print(t.export(args.output, legacy=args.legacy))
