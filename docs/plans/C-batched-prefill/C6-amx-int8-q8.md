# C6. AMX-INT8 for Q8_0 (W8A8)

## 1. Goal and status

Run batched Q8_0 prefill on AMX-INT8 tiles while **preserving the existing Q8_0
semantics**:

- the same per-row, per-group weight scales;
- the same `k_quantize` activation quantization (per token row, per group);
- the same group size.

The result must agree with the AVX-512 Q8 batched path within the kernel budget. W8A8
comes first. W8A32 stays a separate AVX-512 path.

Status: **planned**. Branch: `feature/prefill-c6-amx-int8`.

## 2. Depends on / unblocks

- Depends on: C5 (ladder order, packing pass, budget reporting) and C4.
- Unblocks: C7, D1.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| `k_quantize` | `src/kernels_simd.c:611` | Bitwise identical across ISAs. Applied per token row. |
| W8A8 semantics | `dots_q8_avx512`, `src/kernels_simd.c:163` | Per group: `int32 dot × ws[r,g] × xs[g]`, FMA into FP32. Requires `gs % 64 == 0`. |
| W8A32 | `dots_q8f_avx512`, `src/kernels_simd.c:190` | Stays on AVX-512 |
| Scalar Q8 | `src/kernels_simd.c:511-519` | Oracle |
| INT8 packing helper | C4 `pack_s8_vnni`, B1 `Shard` | Same descriptors for one or two sockets |

## 4. Changes

### 4.1 AVX-512 Q8 batched oracle

First, upgrade C2's per-row Q8 reference to a tiled AVX-512 VNNI batched kernel with
the same per-group semantics. This is both the oracle and the "avx512-q8" ladder rung
in C7.

### 4.2 AMX-INT8 kernel, for each N block × M block

```text
for each group g (K-step of gs = 64 int8 = one tile K-depth):
    C_i32 (16x16 tiles) = 0                       # zero
    _tile_dpbssd(C_i32, A_s8[m, g], B_s8[g, n])   # signed x signed, one group
    store C_i32 → workspace; C_f32[m][n] += C_i32[m][n] * ws[n, g] * xs[m, g]
```

- **Signed × signed.** `tdpbssd` multiplies signed values directly, so the |x|/sign
  trick that VNNI `dpbusd` needs is not needed. The int32 result per group equals the
  AVX-512 path's exactly.
- **Rescale order.** The FP32 rescale per group uses the same operand order as
  `dots_q8_avx512` (`acc += (float)i32 * (ws * xs)`), so results agree within the
  budget. They are not bitwise, because FMA vs separate multiply-add may differ; this
  is recorded.
- **Groups wider than one tile.** For `gs = 128`, accumulate two tile steps before the
  rescale. Any `gs % 64 == 0` is supported. Other group sizes fall back to the AVX-512
  batched path, with the fallback logged.

### 4.3 Packing

The weight int8 values are packed to VNNI quads per shard. Scales stay in their
existing row-major `[row][group]` layout, read per N tile. Packed bytes per socket for
8B:

| | Bytes |
| --- | ---: |
| Packed int8 copy, one socket (**modeled**) | about 7 GiB |
| Retained Q8 for decode | 7.43 GiB |
| Total | about 14.4 GiB |

### 4.4 W8A32

`LLAMA_Q8_ACTIVATIONS=fp32` with `amx` logs `W8A32 prefill uses avx512` and uses the
AVX-512 W8A32 batched kernel. It is never routed through AMX-INT8, since that would
change the activation semantics.

## 5. Local tests

These run on all platforms with scalar oracles; AMX cases run on AMX hosts only.

- **Activations.** `k_quantize` per row in the batched path gives int8 and scales
  bitwise equal to the per-token path.
- **Integer exactness.** Each group's int32 dot from the AMX tiles is exactly equal to
  the scalar int32 dot.
- **Final outputs.** Within `2e-5·Σ|w·x| + 1e-6` of the double reference built from
  the dequantized operands, which is the existing Q8 budget (`tests/kernels.c:126-167`).
  Covers `gs` ∈ {64, 128} and fallback for `gs = 32`.
- **Fixture.** AMX-INT8 forward vs AVX-512 Q8 batched forward within the fixture
  tolerance.

## 6. Cluster run

`cresco8-prefill.sbatch STAGE=check PHASE=C6`. **Estimated** at 60 minutes. It needs
the user's go-ahead to submit.

| Run | Gate or record |
| --- | --- |
| `compare_engines.py` 1B and 8B INT8: AVX-512 Q8 batched (M = 64) vs AMX-INT8 (M = 64), `all` logits | max\|Δlogit\| ≤ 1e-3, argmax 100% |
| `compare_engines.py` AVX-512 Q8 batched vs per-token Q8 | Same gate |
| `validate.py` Q8 recording | max_abs vs HF ≤ per-token Q8 value + 1e-3 |
| Short sweep: 8B Q8, 2048-token prompt, M ∈ {32, 64, 128, 256} | Prefill tok/s recorded |

## 7. Exit criteria

- AMX-INT8 W8A8 agrees with the AVX-512 Q8 batched path and the per-token path.
- Q8 semantics are unchanged, with bitwise activation quantization.
- W8A32 is untouched.
- Packing is reported in the per-socket budget.

## 8. Risks and open questions

- **Rescale overhead.** With `gs = 64`, every tile `dpbssd` (16×16×64 MACs) is
  followed by a 16×16 store, convert and FMA. That overhead can cap AMX-INT8 well below
  the **measured** 69 TOPS library rate. Mitigations, in order:
  1. Keep C tiles for 2×2 blocks and rescale while the next tile op is in flight.
  2. Rescale in registers after `_tile_stored` into an L1-resident buffer.
  3. Measure, and report the achieved fraction.

  Changing the Q8 format (a larger `gs`) is out of scope.

## 9. Not in this phase

- W8A32 on AMX.
- New quantization formats.
- INT8 attention.
