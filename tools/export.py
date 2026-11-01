"""
This script has functions and utilties for model export.
Basically, we have a bunch of versions of the model, and we
want to export them to .bin files to be read from and inferenced in C.

Among the "input" versions of PyTorch files/models:
- Official Llama 2 weights released by Meta
- Huggingface weights available on the hub
- llama2.c (this repo) trained models

Among the "output" versions of .bin files:
- v0: Legacy files of the original llama2.c repo (will eventually be DEPRECATED)
- v1-vN: Improved .bin files with a proper header, cache alignment, etc.

This script aspires to provide all of these conversions.
"""

import argparse
import gzip
import json
import os
import shutil
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import ModelArgs, Transformer, precompute_freqs_cis

# -----------------------------------------------------------------------------
# common utilities


def serialize_fp32(file, tensor):
    """writes one fp32 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = d.astype("<f4", copy=False).tobytes()
    file.write(b)


def serialize_int8(file, tensor):
    """writes one int8 tensor to file that is open in wb mode"""
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = d.tobytes()
    file.write(b)


def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()  # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / torch.where(scale > 0, scale, torch.ones_like(scale))[:, None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr


def validate_model(model):
    p = model.params
    kv = p.n_kv_heads or p.n_heads
    if p.hidden_dim is None:
        p.hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    if min(p.dim, p.hidden_dim, p.n_layers, p.n_heads, kv, p.vocab_size, p.max_seq_len) <= 0:
        raise ValueError("architecture dimensions must be positive")
    if p.dim % p.n_heads or p.n_heads % kv or (p.dim // p.n_heads) % 2:
        raise ValueError("unsupported attention dimensions")
    if not np.isfinite(p.norm_eps) or p.norm_eps <= 0 or not np.isfinite(p.rope_theta) or p.rope_theta <= 1:
        raise ValueError("invalid normalization or RoPE metadata")
    expected = {"tok_embeddings.weight": (p.vocab_size, p.dim),
                "output.weight": (p.vocab_size, p.dim), "norm.weight": (p.dim,)}
    kd = p.dim // p.n_heads * kv
    shapes = {"attention_norm.weight": (p.dim,), "ffn_norm.weight": (p.dim,),
              "attention.wq.weight": (p.dim, p.dim), "attention.wk.weight": (kd, p.dim),
              "attention.wv.weight": (kd, p.dim), "attention.wo.weight": (p.dim, p.dim),
              "feed_forward.w1.weight": (p.hidden_dim, p.dim),
              "feed_forward.w2.weight": (p.dim, p.hidden_dim),
              "feed_forward.w3.weight": (p.hidden_dim, p.dim)}
    for i in range(p.n_layers):
        expected.update({f"layers.{i}.{k}": v for k, v in shapes.items()})
    state = model.state_dict()
    for key, shape in expected.items():
        if key not in state or tuple(state[key].shape) != shape:
            raise ValueError(f"invalid tensor {key}: expected {shape}")
        if not torch.isfinite(state[key]).all():
            raise ValueError(f"nonfinite tensor {key}")
    if p.tie_word_embeddings and not torch.equal(model.tok_embeddings.weight, model.output.weight):
        raise ValueError("tied embedding metadata disagrees with output weights")


def version3_header(model, precision, group_size):
    p = model.params
    scaling = p.rope_scaling or {}
    kind = scaling.get("rope_type", scaling.get("type", "default"))
    if kind not in ("default", "llama3"):
        raise ValueError(f"unsupported RoPE type {kind}")
    factor = scaling.get("factor", 1.0)
    low, high = scaling.get("low_freq_factor", 1.0), scaling.get("high_freq_factor", 4.0)
    original = scaling.get("original_max_position_embeddings", 8192)
    if not all(np.isfinite(v) for v in (factor, low, high)) or factor < 1 or low <= 0 or high <= low or original <= 0:
        raise ValueError("invalid RoPE scaling parameters")
    for value in (p.bos_token_id, p.eos_token_id):
        if not 0 <= value < p.vocab_size:
            raise ValueError("invalid BOS/EOS metadata")
    for value in (p.eot_token_id, p.eom_token_id):
        if not -1 <= value < p.vocab_size:
            raise ValueError("invalid EOT/EOM metadata")
    header = bytearray(256)
    struct.pack_into("<9I", header, 0, 0x616b3432, 3, p.dim, p.hidden_dim, p.n_layers,
                     p.n_heads, p.n_kv_heads or p.n_heads, p.vocab_size, p.max_seq_len)
    struct.pack_into("<3I2fI3f3I4i", header, 36, precision,
                     int(torch.equal(model.tok_embeddings.weight, model.output.weight)), group_size,
                     p.norm_eps, p.rope_theta, int(kind == "llama3"), factor, low, high,
                     original, p.max_seq_len, p.dim // p.n_heads,
                     p.bos_token_id, p.eos_token_id, p.eot_token_id, p.eom_token_id)
    name = p.model_name.encode("utf-8")
    if len(name) > 127:
        raise ValueError("model name must fit 127 UTF-8 bytes")
    header[100:100+len(name)] = name
    return header


# -----------------------------------------------------------------------------
# legacy


def legacy_export(model, filepath, version=0):
    """Original export of llama2.c bin files, i.e. version v0"""
    out_file = open(filepath, "wb")

    # first write out the header
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    p = model.params
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    # legacy format uses negative/positive vocab size as a shared classifier flag
    signed_vocab_size = p.vocab_size if shared_classifier else -p.vocab_size
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack(
        "iiiiiii",
        p.dim,
        hidden_dim,
        p.n_layers,
        p.n_heads,
        n_kv_heads,
        signed_vocab_size,
        p.max_seq_len,
    )
    out_file.write(version3_header(model, 0, 0) if version == 3 else header)

    # next write out the embedding weights
    serialize_fp32(out_file, model.tok_embeddings.weight)

    # now all the layers
    # attention weights
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wq.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wk.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wv.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.attention.wo.weight)
    # ffn weights
    for layer in model.layers:
        serialize_fp32(out_file, layer.ffn_norm.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w1.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w2.weight)
    for layer in model.layers:
        serialize_fp32(out_file, layer.feed_forward.w3.weight)
    # final rmsnorm
    serialize_fp32(out_file, model.norm.weight)
    # freqs_cis
    if version == 0:
        cos, sin = precompute_freqs_cis(p.dim // p.n_heads, p.max_seq_len, p.rope_theta)
        serialize_fp32(out_file, cos)
        serialize_fp32(out_file, sin)

    # final classifier weights
    if not shared_classifier:
        serialize_fp32(out_file, model.output.weight)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


# -----------------------------------------------------------------------------
# new version


def version1_export(model, filepath):
    """
    Export the model weights in full float32 .bin file to be read from C.
    This is same as legacy_export, but with a proper header.
    """
    version = 1

    out_file = open(filepath, "wb")
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack("I", 0x616B3432))
    # 2) write version, which will be int
    out_file.write(struct.pack("i", version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack(
        "iiiiiii",
        p.dim,
        hidden_dim,
        p.n_layers,
        p.n_heads,
        n_kv_heads,
        p.vocab_size,
        p.max_seq_len,
    )
    out_file.write(header)
    # 4) write some other flags
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    out_file.write(struct.pack("B", int(shared_classifier)))
    pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b"\0" * pad)

    # now let's write out all the params
    weights = [
        *[layer.attention_norm.weight for layer in model.layers],
        *[layer.ffn_norm.weight for layer in model.layers],
        model.norm.weight,
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    if not shared_classifier:
        weights.append(model.output.weight)
    for w in weights:
        serialize_fp32(out_file, w)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


def version2_export(model, filepath, group_size=64, version=2):
    """
    Export the model weights in Q8_0 into .bin file to be read from C.
    That is:
    - quantize all weights to symmetric int8, in range [-127, 127]
    - all other tensors (the rmsnorm params) are kept and exported in fp32
    - quantization is done in groups of group_size to reduce the effects of any outliers
    """
    if group_size <= 0 or group_size > 131071:
        raise ValueError("group size must be 1..131071")
    # let's first do some validation for this export type
    while model.params.dim % group_size != 0 or model.layers[0].feed_forward.w1.weight.shape[0] % group_size != 0:
        group_size //= 2
        print(f"BACKOFF: reducing group size to {group_size} to fit hidden_dim")
    weights = [
        model.tok_embeddings.weight,
        *[layer.attention.wq.weight for layer in model.layers],
        *[layer.attention.wk.weight for layer in model.layers],
        *[layer.attention.wv.weight for layer in model.layers],
        *[layer.attention.wo.weight for layer in model.layers],
        *[layer.feed_forward.w1.weight for layer in model.layers],
        *[layer.feed_forward.w2.weight for layer in model.layers],
        *[layer.feed_forward.w3.weight for layer in model.layers],
    ]
    shared_classifier = torch.equal(model.tok_embeddings.weight, model.output.weight)
    if not shared_classifier:
        weights.append(model.output.weight)
    for i, w in enumerate(weights):
        if w.numel() % 4:
            raise ValueError(f"weight {i} requires aligned FP32 scales (element count must be divisible by 4)")
        assert (
            w.numel() % group_size == 0
        ), f"weight {i} has numel {w.numel()}, not a multiple of group_size {group_size}"

    # write
    out_file = open(filepath, "wb")
    # first write out the header. the header will be 256 bytes
    # 1) write magic, which will be uint32 of "ak42" in ASCII
    out_file.write(struct.pack("I", 0x616B3432))
    # 2) write version, which will be int
    out_file.write(struct.pack("i", version))
    # 3) write the params, which will be 7 ints
    p = model.params
    hidden_dim = model.layers[0].feed_forward.w1.weight.shape[0]
    n_kv_heads = p.n_heads if p.n_kv_heads is None else p.n_kv_heads
    header = struct.pack(
        "iiiiiii",
        p.dim,
        hidden_dim,
        p.n_layers,
        p.n_heads,
        n_kv_heads,
        p.vocab_size,
        p.max_seq_len,
    )
    out_file.write(header)
    # 4) write some other flags
    out_file.write(struct.pack("B", int(shared_classifier)))
    out_file.write(struct.pack("i", group_size))  # group size used for quantization
    pad = 256 - out_file.tell()  # pad rest with zeros; tell returns current pos
    assert pad >= 0
    out_file.write(b"\0" * pad)
    if version == 3:
        out_file.seek(0)
        out_file.write(version3_header(model, 1, group_size))
    # now that the header is done, let's write out the model

    # first let's write out all the params that we are keeping in fp32: the norms
    for layer in model.layers:  # attention norms
        serialize_fp32(out_file, layer.attention_norm.weight)
    for layer in model.layers:  # MLP norms
        serialize_fp32(out_file, layer.ffn_norm.weight)
    serialize_fp32(out_file, model.norm.weight)  # final pre-classifier norm

    # now let's write out all the params that we are quantizing to Q8_0
    # note we skip classifier weights, which are shared with the embedding
    ew = []
    for i, w in enumerate(weights):
        # quantize this weight
        q, s, err = quantize_q80(w, group_size)
        # save the int8 weights to file
        serialize_int8(out_file, q)  # save the tensor in int8
        serialize_fp32(out_file, s)  # save scale factors
        # logging
        ew.append((err, w.shape))
        print(
            f"{i+1}/{len(weights)} quantized {tuple(w.shape)} to Q8_0 with max error {err}"
        )

    # print the highest error across all weights, should be very small, e.g. O(~0.001)
    ew.sort(reverse=True)
    print(f"max quantization group error across all weights: {ew[0][0]}")

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


def hf_export(llama_model, filepath, group_size=64, dtype=torch.float32):
    """Generate the pytorch_model.bin state_dict and config.json for HuggingFace"""

    try:
        from transformers.models.llama.configuration_llama import LlamaConfig
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    # Generate LlamaModel state_dict
    hf_state_dict = {}

    # Sometimes we have repeated key values for the heads
    dim = llama_model.params.dim
    num_key_value_heads = llama_model.params.n_kv_heads
    n_rep = llama_model.params.n_heads // num_key_value_heads
    key_value_dim = dim // n_rep

    # HuggingFace needs the weights permuted.
    # See: https://github.com/huggingface/transformers/blob/b132c1703eb1c8bd9dfa4ad6a9be2bfd6ef819e9/src/transformers/models/llama/convert_llama_weights_to_hf.py#L122
    def permute_original(w, n_heads=llama_model.params.n_heads, dim1=dim, dim2=dim):
        return (
            w.view(dim1, dim2)
            .reshape(n_heads, dim1 // n_heads // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    # Transfer weights from llama model to the HF state dictionary format
    hf_state_dict["model.embed_tokens.weight"] = (
        llama_model.tok_embeddings.weight.clone().to(dtype)
    )
    hf_state_dict["model.norm.weight"] = llama_model.norm.weight.clone().to(dtype)

    # Add each layer's weights to the HF state dictionary
    for i, layer in enumerate(llama_model.layers):
        layer_id = layer.layer_id
        hf_state_dict[f"model.layers.{i}.input_layernorm.weight"] = (
            llama_model.layers[layer_id].attention_norm.weight.clone().to(dtype)
        )
        hf_state_dict[f"model.layers.{i}.self_attn.q_proj.weight"] = permute_original(
            llama_model.layers[layer_id].attention.wq.weight.clone()
        ).to(dtype)
        hf_state_dict[f"model.layers.{i}.self_attn.k_proj.weight"] = permute_original(
            llama_model.layers[layer_id].attention.wk.weight.clone(),
            num_key_value_heads,
            key_value_dim,
            dim,
        ).to(dtype)
        hf_state_dict[f"model.layers.{i}.self_attn.v_proj.weight"] = (
            llama_model.layers[layer_id].attention.wv.weight.clone().to(dtype)
        )
        hf_state_dict[f"model.layers.{i}.self_attn.o_proj.weight"] = (
            llama_model.layers[layer_id].attention.wo.weight.clone().to(dtype)
        )
        hf_state_dict[f"model.layers.{i}.post_attention_layernorm.weight"] = (
            llama_model.layers[layer_id].ffn_norm.weight.clone().to(dtype)
        )
        hf_state_dict[f"model.layers.{i}.mlp.gate_proj.weight"] = (
            llama_model.layers[layer_id].feed_forward.w1.weight.clone().to(dtype)
        )
        hf_state_dict[f"model.layers.{i}.mlp.down_proj.weight"] = (
            llama_model.layers[layer_id].feed_forward.w2.weight.clone().to(dtype)
        )
        hf_state_dict[f"model.layers.{i}.mlp.up_proj.weight"] = (
            llama_model.layers[layer_id].feed_forward.w3.weight.clone().to(dtype)
        )

    # llama2.c usually uses tied weights -> reference the embed_tokens.weights instead
    hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

    # We check that the embeddings are tied, else use manual output weights
    _embeddings_are_tied: bool = torch.equal(
        llama_model.tok_embeddings.weight, llama_model.output.weight
    )
    if not _embeddings_are_tied:
        hf_state_dict["lm_head.weight"] = llama_model.output.weight.clone().to(dtype)

    # Generate LlamaConfig (seen in transformers.models.llama.configuration_llama)

    # Extract necessary attributes from llama.c model
    vocab_size = llama_model.params.vocab_size
    hidden_size = llama_model.params.dim
    intermediate_size = llama_model.layers[0].feed_forward.w1.weight.shape[0]
    num_hidden_layers = llama_model.params.n_layers
    num_attention_heads = llama_model.params.n_heads
    num_key_value_heads = llama_model.params.n_kv_heads
    max_position_embeddings = llama_model.params.max_seq_len
    rms_norm_eps = llama_model.params.norm_eps

    # TODO check values for:
    # pretraining_tp, initializer_range, use_cache,
    # rope_theta, and rope_scaling.

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        tie_word_embeddings=_embeddings_are_tied,
        rope_theta=llama_model.params.rope_theta,
        rope_scaling=llama_model.params.rope_scaling,
        bos_token_id=llama_model.params.bos_token_id,
        eos_token_id=llama_model.params.eos_token_id,
        # Text-only architecture
        architectures=["LlamaForCausalLM"],
        hidden_act="silu",
    )

    # Save files in directory filepath
    # First make the directory if it doesn't exist
    os.makedirs(filepath, exist_ok=True)

    # Save the state dictionary in .bin format, and config as .json
    torch.save(hf_state_dict, os.path.join(filepath, "pytorch_model.bin"))
    config.save_pretrained(filepath)


# -----------------------------------------------------------------------------
# Load / import functions


def load_checkpoint(checkpoint):

    # load the provided model checkpoint
    checkpoint_dict = torch.load(checkpoint, map_location="cpu")
    gptconf = ModelArgs(**checkpoint_dict["model_args"])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def load_meta_model(model_path):
    params_path = os.path.join(model_path, "params.json")
    with open(params_path) as f:
        params = json.load(f)
        print(params)

    model_paths = sorted(list(Path(model_path).glob("consolidated.*.pth")))
    if not model_paths:
        raise ValueError("no consolidated.*.pth checkpoints found")
    models = [torch.load(p, map_location="cpu", weights_only=True) for p in model_paths]

    def concat_weights(models):
        state_dict = {}
        for name in list(models[0]):
            tensors = [model[name] for model in models]
            if len(tensors) == 1 or len(tensors[0].shape) == 1:
                state_dict[name] = tensors[0]
                continue
            is_axis_1 = (
                name.startswith("tok_embeddings.")
                or name.endswith(".attention.wo.weight")
                or name.endswith(".feed_forward.w2.weight")
            )
            axis = 1 if is_axis_1 else 0
            state_dict[name] = torch.cat(tensors, dim=axis)
            for model in models:
                del model[name]
        return state_dict

    state_dict = concat_weights(models)
    del models

    # set ModelArgs
    config = ModelArgs()
    config.dim = params["dim"]
    config.n_layers = params["n_layers"]
    config.n_heads = params["n_heads"]
    config.n_kv_heads = params.get("n_kv_heads") or params["n_heads"]
    config.multiple_of = params["multiple_of"]
    config.norm_eps = params["norm_eps"]

    config.vocab_size = state_dict["tok_embeddings.weight"].shape[0]
    config.hidden_dim = state_dict["layers.0.feed_forward.w1.weight"].shape[0]
    config.rope_theta = params.get("rope_theta", 500000.0)
    config.rope_scaling = params.get("rope_scaling")
    if params.get("use_scaled_rope") and not config.rope_scaling:
        raise ValueError("Meta use_scaled_rope requires explicit rope_scaling metadata; use an HF config/export")
    config.max_seq_len = params.get("max_seq_len", params.get("max_position_embeddings", 8192))
    config.tie_word_embeddings = params.get("tie_word_embeddings", False)
    config.model_name = Path(model_path).name

    # create a new Transformer object and set weights
    with torch.device("meta"):
        model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(state_dict["tok_embeddings.weight"])
    model.norm.weight = nn.Parameter(state_dict["norm.weight"])

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(
            state_dict[f"layers.{i}.attention_norm.weight"]
        )
        layer.attention.wq.weight = nn.Parameter(
            state_dict[f"layers.{i}.attention.wq.weight"]
        )
        layer.attention.wk.weight = nn.Parameter(
            state_dict[f"layers.{i}.attention.wk.weight"]
        )
        layer.attention.wv.weight = nn.Parameter(
            state_dict[f"layers.{i}.attention.wv.weight"]
        )
        layer.attention.wo.weight = nn.Parameter(
            state_dict[f"layers.{i}.attention.wo.weight"]
        )
        layer.ffn_norm.weight = nn.Parameter(state_dict[f"layers.{i}.ffn_norm.weight"])
        layer.feed_forward.w1.weight = nn.Parameter(
            state_dict[f"layers.{i}.feed_forward.w1.weight"]
        )
        layer.feed_forward.w2.weight = nn.Parameter(
            state_dict[f"layers.{i}.feed_forward.w2.weight"]
        )
        layer.feed_forward.w3.weight = nn.Parameter(
            state_dict[f"layers.{i}.feed_forward.w3.weight"]
        )

    # final classifier
    model.output.weight = nn.Parameter(state_dict.get("output.weight", state_dict["tok_embeddings.weight"]))
    model.eval()
    return model


def load_hf_model(model_path):

    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        print("Error: transformers package is required to load huggingface models")
        print("Please run `pip install transformers` to install it")
        return None

    if not Path(model_path).is_dir():
        raise ValueError("HF conversion requires a local snapshot directory")
    hc = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    if getattr(hc, "quantization_config", None):
        raise ValueError("external quantization formats cannot be imported; provide original FP32/FP16/BF16 weights")
    if hc.model_type != "llama" or hc.hidden_act != "silu" or getattr(hc, "attention_bias", False) or getattr(hc, "mlp_bias", False):
        raise ValueError("only bias-free text-only Llama with SwiGLU is supported")
    if (getattr(hc, "head_dim", None) or hc.hidden_size // hc.num_attention_heads) != hc.hidden_size // hc.num_attention_heads:
        raise ValueError("independent head_dim is not supported")
    if getattr(hc, "partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("partial rotary embeddings are not supported")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, config=hc, torch_dtype=torch.float32,
        local_files_only=True, attn_implementation="eager", trust_remote_code=False)
    hf_dict = hf_model.state_dict()

    # convert LlamaConfig to ModelArgs
    config = ModelArgs()
    config.dim = hf_model.config.hidden_size
    config.n_layers = hf_model.config.num_hidden_layers
    config.n_heads = hf_model.config.num_attention_heads
    config.n_kv_heads = hf_model.config.num_key_value_heads
    config.vocab_size = hf_model.config.vocab_size
    config.hidden_dim = hf_model.config.intermediate_size
    config.norm_eps = hf_model.config.rms_norm_eps
    config.max_seq_len = hf_model.config.max_position_embeddings

    config.rope_theta = hc.rope_theta
    config.rope_scaling = hc.rope_scaling
    config.tie_word_embeddings = hc.tie_word_embeddings
    config.bos_token_id = hc.bos_token_id
    eos = hc.eos_token_id
    config.eos_token_id = eos[0] if isinstance(eos, list) else eos
    tokenizer_path = Path(model_path) / "tokenizer.json"
    specials = {}
    if tokenizer_path.exists():
        tokenizer_data = json.loads(tokenizer_path.read_text())
        specials = {t["content"]: t["id"] for t in tokenizer_data["added_tokens"] if t["special"]}
    elif hc.vocab_size >= 128000:
        raise ValueError("real Llama conversion requires tokenizer.json for special-token metadata")
    config.bos_token_id = specials.get("<|begin_of_text|>", config.bos_token_id)
    config.eos_token_id = specials.get("<|end_of_text|>", config.eos_token_id)
    config.eot_token_id = specials.get("<|eot_id|>", -1)
    config.eom_token_id = specials.get("<|eom_id|>", -1)
    source = Path(str(model_path))
    cached_repo = source.parent.parent.name
    config.model_name = cached_repo[len("models--"):].replace("--", "/") if cached_repo.startswith("models--") else source.name
    # Allocate shape-only modules; imported tensors supply all actual storage.
    with torch.device("meta"):
        model = Transformer(config)

    model.tok_embeddings.weight = nn.Parameter(hf_dict["model.embed_tokens.weight"])
    model.norm.weight = nn.Parameter(hf_dict["model.norm.weight"])

    # huggingface permutes WQ and WK, this function reverses it
    def permute_reverse(w, n_heads=config.n_heads, dim1=config.dim, dim2=config.dim):
        return (
            w.view(n_heads, 2, dim1 // n_heads // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    for layer in model.layers:
        i = layer.layer_id
        layer.attention_norm.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.input_layernorm.weight"]
        )
        layer.attention.wq.weight = nn.Parameter(
            permute_reverse(hf_dict[f"model.layers.{i}.self_attn.q_proj.weight"])
        )
        layer.attention.wk.weight = nn.Parameter(
            permute_reverse(hf_dict[f"model.layers.{i}.self_attn.k_proj.weight"], config.n_kv_heads, config.dim // config.n_heads * config.n_kv_heads)
        )
        layer.attention.wv.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.self_attn.v_proj.weight"]
        )
        layer.attention.wo.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.self_attn.o_proj.weight"]
        )
        layer.ffn_norm.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.post_attention_layernorm.weight"]
        )
        layer.feed_forward.w1.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.mlp.gate_proj.weight"]
        )
        layer.feed_forward.w2.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.mlp.down_proj.weight"]
        )
        layer.feed_forward.w3.weight = nn.Parameter(
            hf_dict[f"model.layers.{i}.mlp.up_proj.weight"]
        )

    # final classifier
    model.output.weight = nn.Parameter(hf_dict["lm_head.weight"])
    model.eval()
    return model


# -----------------------------------------------------------------------------
# API entrypoint


def model_export(model, filepath, version, dtype=torch.float32):
    """
    Versions docs:
    v-1:huggingface export, i.e. intended for use outside of this repo, in HF
    v0: legacy llama2.c float format, DEPRECATED
    v1: float32 export
    v2: int8 quantized Q8_0 export, similar to llama.cpp, in groups
    # TODO: add dtype export support for other versions (?)
    """
    validate_model(model)
    if version in (0, 1, 2) and (model.params.rope_scaling or model.params.rope_theta != 500000.0 or model.params.norm_eps != 1e-5):
        raise ValueError("legacy formats cannot represent model metadata; use --version 3")
    if version == 3:
        version3_header(model, 0, 0)  # validate all metadata before opening output
        legacy_export(model, filepath, version=3)
    elif version == 0:
        legacy_export(model, filepath)
    elif version == 1:
        version1_export(model, filepath)
    elif version == 2:
        version2_export(model, filepath)
    elif version == -1:
        hf_export(model, filepath, dtype=dtype)
    else:
        raise ValueError(f"unknown version {version}")


def torchscript_export(model, filepath, zero_params=False, gzip_output=False):
    """
    (This was submitted via a PR earlier. Leaving it here, but "orphaned" for now)
    Saves the model as a TorchScript.
    The resulting file can be loaded in C++ code and then used for training or
    inference with:
        #include <torch/script.h>
        torch::jit::Module module = torch::jit::load("model.pt")
    Note that the serialized model includes the initial parameters and with the default
    ModelArgs the file is 59M and gzips down to 55M. If you want to serialize/distribute
    the model parameters separately you can zero out the parameters before saving it and
    it will gzip down to 780K.
    """

    # If requested zero params before saving the model. This is useful in
    # conjunction with gzip_output.
    if zero_params:
        for p in model.parameters():
            p.detach().zero_()

    torch.jit.save(torch.jit.script(model), filepath)

    if gzip_output:
        with open(filepath, "rb") as f_in:
            with gzip.open(f"{filepath}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.unlink(filepath)


# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument(
        "--version", default=3, type=int, help="the version to export with"
    )
    parser.add_argument(
        "--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32"
    )
    parser.add_argument("--int8", action="store_true", help="Q8_0 payload for version 3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    group.add_argument("--meta-llama", type=str, help="meta llama model path")
    group.add_argument("--hf", type=str, help="huggingface model path")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)
    elif args.meta_llama:
        model = load_meta_model(args.meta_llama)
    elif args.hf:
        model = load_hf_model(args.hf)

    if model is None:
        parser.error("Can't load input model!")

    # export
    if args.int8:
        if args.version != 3:
            parser.error("--int8 requires --version 3")
        validate_model(model)
        version3_header(model, 1, 64)
        version2_export(model, args.filepath, version=3)
    else:
        if args.dtype != "fp32" and args.version != -1:
            parser.error("C checkpoints require --dtype fp32 (or --int8)")
        model_export(model, args.filepath, args.version, dtype)
