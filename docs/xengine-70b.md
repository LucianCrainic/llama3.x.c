# Llama 3.1 70B across three CPU inference engines

> **Measured on an older tree (the earlier 70B comparison work, before the prefill, Wo/W2 and gs32/gs64 kernel work); numbers are not comparable with current-tree measurements.**

Numbers are measured. [The validation record](xengine-70b-validation.md) lists the jobs. The llama3.x.c Q8 DDR rows were re-measured on the current tree (`ee24659`, jobs 5929406 and 5929407); the llama3.x.c Q8 HBM rows could not be re-run there and stay STALE. Everything else was not re-run on the current tree.

This comparison pins the same source snapshot, input token IDs, output length and CPU resources. All three engines run FP32 on DDR. llama3.x.c and llama.cpp also run identical numeric Q8_0 weights.

## Setup

- **Model.** Llama 3.1 70B Instruct, NousResearch mirror revision `d50656ee28e2c2906d317cbbb6fcb55eb4055a84`. The FP32 conversions widen the original BF16 values exactly. llama3.x.c loads v3, llama.cpp loads an F32 GGUF, and vLLM widens the same snapshot at load. [Pinned snapshot](https://huggingface.co/NousResearch/Meta-Llama-3.1-70B-Instruct/tree/d50656ee28e2c2906d317cbbb6fcb55eb4055a84)
- **Engine versions.** llama3.x.c ran the pre-port revision of jobs **5910133**, **5910211** and **5913407** (the hybrid research tree on top of `1bc9189`) with the project's default Release flags, AVX-512 dispatch and grouped decode attention. llama.cpp uses `b1b770d90d94cad9e9ea5b993bdbc383a522fb28`, built in Release mode with GCC Toolset 14, `GGML_NATIVE=ON`, `GGML_OPENMP=OFF` and `GGML_BLAS=OFF`. vLLM is stock `0.34.0+cpu` from the pinned image. Build and allocation details live in the raw run logs.
- **Node and memory.** Exclusive CRESCO8 Flat/Quadrant node, two Xeon Max 9480 sockets, 55 cores per socket available to each engine. The three-way track uses DDR and FP32 weights, activations and K/V storage. The C and vLLM engines use socket-local tensor parallelism. The llama.cpp CPU worker pool is pinned across the same cores and uses memory interleaving across the chosen tier.
- **Requests.** The 547-ID input is the concatenated 512-position warm-up and 35-token prompt from the hybrid pilot. The 2,113-ID input is the AMX campaign prompt. Every engine receives each array as one request, generates 128 greedy tokens, ignores EOS, caps the context at 8,192 and uses a prefill batch limit of 256. One warm-up request per shape precedes three measurements. Prefix caching is disabled for vLLM and each native request starts with an empty logical cache.
- **Timing.** TTFT starts at request submission with token IDs and stops at the first greedy token. Decode is 127 divided by the time from first to last generated token. The native drivers use the same in-process timer. vLLM uses local streaming HTTP, so its client timings include local server and transport overhead. Startup and checkpoint conversion are outside these measurements.

The native INT8 comparison uses a single Q8_0 GGUF. [The repacker](../tools/xengine70/repack_q8.py) copies the 32-value blocks exactly, widens their FP16 scales to FP32 and preserves the RoPE ordering used by both engines. The physical scale storage and kernel implementations differ. Both set the KV cache to FP32. On the pre-port tree the accelerated W8A8 matmul kernels required groups divisible by 64, so this exact 32-value checkpoint ran on the portable matmuls (the old C-engine Q8 rows are therefore STALE). The current tree has the gs32 AMX-INT8 prefill and VNNI decode kernels ([gs32-amx.md](gs32-amx.md)), which the re-run of the DDR rows below used; the hybrid study used the native group-64 checkpoint, which selected the VNNI decode and AMX prefill kernels even then. Stock vLLM CPU supports compressed-tensors for W8A8, and its GGUF implementation is GPU-targeted. [vLLM CPU quantization support](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/index.html) · [vLLM quantization matrix](https://docs.vllm.ai/en/latest/features/quantization/)

## Results

### FP32 on DDR

All timings are medians [min–max] across the three measured requests. Rates use 127 decode steps after the first token.

| Engine | Prompt tokens | TTFT (s) | Decode (tok/s) |
| --- | ---: | ---: | ---: |
| llama3.x.c | 547 | 13.625 [13.624–13.638] | 1.766 [1.765–1.766] |
| vLLM | 547 | 10.018 [10.014–10.020] | 1.680 [1.680–1.681] |
| llama.cpp | 547 | 22.704 [22.702–24.684] | 1.027 [1.014–1.036] |
| llama3.x.c | 2,113 | 52.996 [52.995–52.998] | 1.757 [1.757–1.757] |
| vLLM | 2,113 | 37.782 [37.774–37.784] | 1.671 [1.670–1.671] |
| llama.cpp | 2,113 | 88.630 [88.559–88.707] | 1.032 [1.029–1.046] |

The C engine decoded 5.1–5.2% faster than vLLM and 1.70–1.72× as fast as llama.cpp on these requests. The C engine's time to first token was 36.0–40.3% longer than vLLM's and 40.0–40.2% shorter than llama.cpp's. Median submission-to-last-token time for the 547-token prompt was 85.53 s for llama3.x.c, 85.59 s for vLLM and 146.42 s for llama.cpp. The corresponding 2,113-token medians were 125.27 s, 113.79 s and 211.61 s. The entire 128-token greedy continuation agreed across all three engines and every measured repeat on both shapes.

Jobs **5910133** (llama3.x.c), **5910177** (vLLM) and **5912354** (llama.cpp) ran on `cresco8-hbm14`. Resident memory samples place 271,904 MiB for the C process, 291,576 MiB across the two vLLM workers and 274,465 MiB for llama.cpp entirely in DDR. The C engine verified the active KV pages after the run. llama.cpp logged 8,192 cells and FP32 K and V. The vLLM API-client TTFT averaged 7.4 ms above the corresponding server metric across the eight requests. Slurm job 5910177 was stopped during container shutdown after all eight validated traces, the server metrics and worker NUMA samples were written.

### Q8_0 with identical quantized weights

**Re-measured on the current tree (`ee24659`, engine defaults, no `LEGACY`; jobs 5929406 gs32 and 5929407 gs64, node `cresco8-hbm14`) for the llama3.x.c DDR rows. The llama3.x.c HBM rows are still STALE (pre gs32 AMX/VNNI, measured on the portable group-32 kernels) because that cell does not run on the current tree, see below. The llama.cpp rows are unaffected.**

| Memory | Engine | Prompt tokens | TTFT (s) | Decode (tok/s) |
| --- | --- | ---: | ---: | ---: |
| DDR | llama3.x.c (current tree, gs32) | 547 | 2.689 [2.686–2.693] | 5.928 [5.928–5.931] |
| DDR | llama.cpp | 547 | 5.928 [5.820–6.069] | 3.100 [3.066–3.131] |
| DDR | llama3.x.c (current tree, gs32) | 2,113 | 10.222 [10.200–10.236] | 5.755 [5.749–5.760] |
| DDR | llama.cpp | 2,113 | 23.716 [23.319–23.793] | 3.037 [3.033–3.068] |
| HBM | llama3.x.c (STALE) | 547 | 84.484 [84.484–84.499] | 5.343 [5.341–5.346] |
| HBM | llama.cpp | 547 | 17.068 [16.129–17.354] | 2.704 [2.683–2.733] |
| HBM | llama3.x.c (STALE) | 2,113 | 326.997 [326.994–327.051] | 5.290 [5.289–5.291] |
| HBM | llama.cpp | 2,113 | 67.208 [65.885–68.249] | 2.639 [2.626–2.672] |

Re-run (DDR, same 547- and 2,113-token prompts, 128 greedy tokens, one warm-up and three measured requests, two sockets of 55 threads, prefill batch 256, the same group-32 checkpoint as before): the C engine's TTFT fell from 84.236 s to 2.689 s (547 tokens, 31.3× shorter) and from 326.208 s to 10.222 s (2,113 tokens, 31.9× shorter). Decode rose from 5.068 to 5.928 tok/s (+17.0%) and from 4.489 to 5.755 tok/s (+28.2%). Against llama.cpp on DDR the C engine's TTFT is now 0.45× and 0.43× of llama.cpp's (was 14.2× and 13.8× as long), and it decodes 1.91× and 1.89× as fast (was 1.64× and 1.48×). The startup log confirms `LLAMA_KERNELS=auto` with AMX for the batched prefill. The old values in this paragraph came from the pre-port tree (job 5910211) and are kept for reference only.

The same two workloads on the native group-64 checkpoint (a different file with different quantization, so not the identical-weights comparison) gave TTFT 2.154 s [2.150–2.163] and 8.065 s [8.063–8.072], decode 6.216 tok/s [6.210–6.228] and 6.011 tok/s [6.007–6.030] (job 5929407).

The HBM cell does not run: the current engine places 83.0 GB (gs32; 78.6 GB for gs64) per socket, against the 56 GiB (60.13 GB) per-socket HBM budget, and with strict placement it stops at startup ("needs ... bytes of hbm (22,870,310,912 over its 60,129,542,144-byte budget)", 18,493,263,872 over for gs64). The old tree held 43 GB per socket on the same file, so the current per-socket footprint is about 1.9× the old one (the cause of the larger footprint was not investigated here); the 64 GiB of physical HBM per socket would not hold it either. The HBM rows therefore keep the old values.

With the old (STALE) C-engine kernels, HBM raised the C engine's decode rate by 5.4% at 547 tokens and 17.8% at 2,113 tokens. Median TTFT moved by under 1 s for each prompt. In the tested llama.cpp interleaved configuration, HBM reduced decode throughput by 12.8–13.1% and increased TTFT by 2.83–2.88×. These are results for the 110-worker pool with interleaving over both memory nodes of the selected tier.

The current-tree C engine's 128 output IDs equal the pre-port ones on all four requests of both workloads for the gs32 checkpoint (job 5929406 against job 5910211), so the numerics did not change for this file; the gs64 checkpoint gives the same long continuation but diverges from the gs32 one at the fourth ID of the short prompt (a different checkpoint, not a kernel change). Both engines matched their own output IDs across repeats and memory tiers. Across engines the output IDs match for all 128 long-prompt tokens. The short continuations share the first three IDs. The workloads use identical prompt IDs, 128 greedy output steps and the same attention depth at every step.

Job **5910211** completed both DDR cells and the C HBM cell on `cresco8-hbm14`. It reached its time limit during the llama.cpp HBM cell. Job **5913407** repeated that cell on the same node with all eight requests. The DDR resident samples were 78,475 MiB for the C process and 85,109 MiB for llama.cpp, all on DDR. HBM resident samples were 78,462 MiB and 85,106 MiB, respectively, with under 13 MiB of DDR residency in either process. The C engine's final active KV page checks passed in both placements.

## Reproduction

The [Slurm harness](../scripts/xengine70.sbatch) records the commands, environment metadata, memory placement and request traces. The source snapshot, prompt IDs, launcher and run metadata of the jobs above are under `70b-engine-comparison-20261124` in cluster scratch. Conversion jobs **5910127**, **5910128** and **5910181** produced the F32 v3, GGUF and repacked Q8_0 checkpoints. The [summarizer](../tools/xengine70/summarize.py) checks the expected request lengths and uses ranges across the three measured repeats.

Run the checkpoint conversions in an allocation with enough memory and scratch space. Here `SNAPSHOT` names the pinned HF directory and `CPP` names the pinned llama.cpp checkout.

```bash
python3 tools/export.py fp32-v3.bin --hf "$SNAPSHOT" --version 3
python3 "$CPP/convert_hf_to_gguf.py" "$SNAPSHOT" --outtype q8_0 --outfile q8_0.gguf
python3 "$CPP/convert_hf_to_gguf.py" "$SNAPSHOT" --outtype f32 --outfile f32.gguf
PYTHONPATH="$CPP/gguf-py" python3 tools/xengine70/repack_q8.py \
  q8_0.gguf int8-v3.bin q8-gguf-v3.bin
```

`int8-v3.bin` in the last command is the hybrid study's native group-64 checkpoint. The repacker reads its 256-byte model header.

Copy the [prompt fixtures](../tools/xengine70/prompts/) to `$RUN_ROOT/prompts` and put the checkpoints in `$RUN_ROOT/models`. The native drivers read the corresponding whitespace-delimited files. Both representations were checked token by token.

```bash
python3 - "$RUN_ROOT/prompts" <<'PY'
import json
from pathlib import Path
import sys

for path in Path(sys.argv[1]).glob("*.ids.json"):
    tokens = json.loads(path.read_text())
    path.with_suffix(".txt").write_text(" ".join(map(str, tokens)) + "\n")
PY
```

The JSON SHA-256 values are `48667d28324cb4927d4d2a6b5411d4c3a717b8a028c48e19645e6ec6748b3d24` (547 tokens) and `412858a5ba9116f557bdafd6289c47cff874f4d0f400edce5724e6fe2c9a092d` (2,113 tokens).

Submit each mode sequentially on the same exclusive Flat node. Set `MODE` to `c-fp32`, `cpp-fp32`, `vllm-fp32`, `c-q8`, `cpp-q8`, `c-q8-hbm` or `cpp-q8-hbm`. Use the pinned CPU SIF for `VLLM_IMAGE` and a checkout of this tree for `ROOT` (`make xengine70-fp32` and `make xengine70-q8` build the native drivers; the script runs them). The C modes leave the engine at its defaults except for the socket and thread counts, the prefill batch, the placement and the page verification; set `LEGACY=1` for the kernels and attention of the jobs above.

```bash
sbatch --nodelist "$FLAT_NODE" "$ROOT/scripts/xengine70.sbatch" \
  "$ROOT" "$RUN_ROOT/$MODE" "$SNAPSHOT" "$CPP" "$VLLM_IMAGE" "$RUN_ROOT/prompts" "$MODE"
```
