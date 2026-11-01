# F1. Validated tensor manifest

## 1. Goal and status

Replace pointer-bumping weight mapping with a validated **tensor manifest**: one
descriptor per tensor, holding file offset, shape, dtype, scale offset, layer and role.
The manifest is built once, after `checkpoint_map` finishes its existing checks.

Several later phases consume it:

| Phase | What it builds from the manifest |
| --- | --- |
| A4 | Owned weight copies |
| B1 | Socket shards |
| C1, C5, C6 | Batched and packed (BF16, INT8 VNNI) views |

All of them describe the same tensors in the same way, so there is no second
scheduler or descriptor family (see the user decision in the [README](../README.md) §5).

Inference behaviour does not change: weights are still read from the file mapping, at
exactly the pointers used today.

Status: **implemented, local tests pass** (not yet merged). Branch:
`feature/found-f1-tensor-manifest`.

Implementation notes (deviations from the proposal above):

- `checkpoint_map` takes a `TensorManifest *` and builds it after its own size
  arithmetic; `manifest_build`'s end offset is compared with that arithmetic and a
  mismatch fails. `tensor_bytes` is now non-static so the builder reuses it.
- The oracle test is `tests/manifest.c` (`test-manifest`), driven from
  `tests/test_runtime.py::test_manifest_matches_layouts`, because the old pointer
  code is static inside `run.c`/`runq.c`.
- The trace dump is enabled with the existing `--log-level trace`; there is no
  `LLAMA_LOG` variable.

## 2. Depends on / unblocks

- Depends on: nothing.
- Unblocks: A4, B1, C1, and indirectly every phase that packs weights.

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| `checkpoint_map` | `src/runtime.c:131` | Keeps every check: ABI and endianness (:133-137), header formats v3/v2/legacy (:174-223), `validate_config` (:100), exact-size arithmetic (:230-249) |
| `tensor_bytes` | `src/runtime.c:122` | FP32 vs Q8 byte size (q bytes + FP32 scales, no padding) |
| `checked_mul` / `checked_add` | `src/runtime.c:57-66` | Overflow checks for every offset |
| FP32 file order | `src/run.c:87` `memory_map_weights` | embedding, rms_att[L], wq, wk, wv, wo, rms_ffn[L], w1, w2, w3, rms_final, legacy RoPE (format 0), wcls |
| Q8 file order | `src/runq.c:151` `memory_map_weights`, `init_quantized_tensors` :136 | norms first, then q_tokens, then per-type `[L]` arrays of `q` followed by `s` |

## 4. Changes

New `include/shards.h` and `src/shards.c`. This phase adds only the manifest part;
B1 adds the shard part.

```c
/* Proposed. */
typedef enum { DT_F32, DT_Q8 } TensorDtype;
typedef enum {
  T_EMBED, T_RMS_ATT, T_WQ, T_WK, T_WV, T_WO, T_RMS_FFN, T_W1, T_W2, T_W3,
  T_RMS_FINAL, T_ROPE_LEGACY, T_WCLS, T_ROLE_COUNT
} TensorRole;
typedef struct {
  TensorRole role;
  int layer;              /* -1 for global tensors */
  int rows, cols;         /* W[out][in]: rows = out, cols = in */
  TensorDtype dtype;
  int group_size;         /* 0 for F32 */
  size_t offset;          /* bytes from file start: values */
  size_t scale_offset;    /* bytes from file start: Q8 scales, else 0 */
  size_t bytes;           /* values + scales */
  int alias_of;           /* index of shared tensor (tied wcls), else -1 */
} TensorDesc;
typedef struct { TensorDesc *t; int count; size_t file_bytes; } TensorManifest;

void manifest_build(TensorManifest *m, const Config *p, int quantized, int gs, size_t header_bytes);
const TensorDesc *manifest_find(const TensorManifest *m, TensorRole role, int layer);
void manifest_free(TensorManifest *m);
```

The construction rules:

- **One builder for both formats.** `manifest_build` describes the layouts of both
  `run` and `runq`. Its final offset must equal the exact size that `checkpoint_map`
  already computes; any mismatch is a hard failure, not a warning.
- **Tied classifier.** A tied `wcls` gets `alias_of` pointing at `T_EMBED`. B1 uses this
  to shard the shared tensor instead of copying it (study §5.1).
- **Storage.** The manifest is allocated as `MEM_METADATA`.
- **Old mapping functions become views.** Both `memory_map_weights` functions build
  their pointers by `data + desc->offset`. `runq.c` keeps its `QuantizedTensor` arrays,
  and `init_quantized_tensors` reads offsets from the manifest.
- **Diagnostics.** `LLAMA_LOG=trace` dumps the manifest as a table, which helps with
  later packing bugs.

## 5. Local tests

These all run on macOS.

- `tests/unit.c`, new case `manifest_matches_mapping`:
  - For both fixture checkpoints (FP32 and INT8, shared and unshared), every
    manifest-derived pointer equals the pointer the old code computed.
  - Keep the old pointer-bumping code in the test only, as the oracle.
- `test_runtime.py`: the 48-value logit baseline (`tests/baseline_logits.json`) is
  unchanged, **bitwise**.
- The truncated and oversized checkpoint tests still fail with the same messages.
- `make MODE=sanitize test` shows no leaks; `memory_current` returns to zero.

## 6. Cluster run

None. The change is layout-only and fully covered by local tests. F3's first `check`
job also exercises it on 1B and 8B as part of `make test` and `tools/validate.py`.

## 7. Exit criteria

- The manifest reproduces every current weight pointer for FP32 and INT8, shared and
  unshared, and for formats legacy, v2 and v3.
- Logits are bitwise unchanged on the fixtures.
- `checkpoint_map` keeps all its error messages. No new I/O path exists yet.

## 8. Risks and open questions

- **Legacy format 0 RoPE table.** It sits between `rms_final` and `wcls` in FP32 files
  only. The manifest must include it so later offsets are right, even though it is
  never used.
- **Q8 `q_tokens` order.** The INT8 v2/v3 layout puts the norms *before* `q_tokens`
  (`src/runq.c:151-179`). The builder needs one function per precision, not one shared
  ordering with a flag.

## 9. Not in this phase

- Copying weights (A4).
- Splitting tensors into shards (B1).
- Packing for GEMM or AMX (C2, C5, C6).
- Prepacked checkpoint formats.
