# A4. Owned weight loader

## 1. Goal and status

Load the checkpoint into engine-owned, explicitly bound anonymous regions instead of
running from the read-only file mapping. On one socket, every frequently read weight
goes to the selected node. The untied input embedding goes to DDR. Inference math is
identical, and logits must be **bitwise** equal to `LLAMA_PLACEMENT=file`.

Status: **planned**. Branch: `feature/numa-a4-owned-loader`.

## 2. Depends on / unblocks

- Depends on: F1 (manifest), A2 (regions) and A3 (verification).
- Unblocks: A5, A6, B1 (B1 extends this loader with slicing and packing), and C5/C6
  (packed copies are produced by the same pass).

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| `checkpoint_map` | `src/runtime.c:131` | All validation stays. Its `mmap` (:253) becomes either the source for copying or the `file` mode itself. |
| Manifest | F1 | Offsets and sizes per tensor |
| FP32 views | `src/run.c:87` | Point views at owned copies instead of the mapping |
| Q8 views and embedding expansion | `src/runq.c:151-179`; dequantization at :166, scalar `dequantize` at :116 | Replace the full-table expansion with row-wise dequantization |
| Classifier tie | `src/run.c:87` (shared ⇒ `wcls = token_embedding_table`) | The tie must be preserved |

## 4. Changes

### 4.1 Placement policy by role

This applies study §5.1 to one socket:

| Data | `LLAMA_PLACEMENT=hbm` | `=ddr` | `=file` |
| --- | --- | --- | --- |
| Layer matrices, Q8 scales, classifier | socket's HBM node | socket's DDR node | file mapping (today) |
| Norm weights, legacy RoPE | HBM, small copy | DDR | file mapping |
| Untied input embedding | socket's DDR node | DDR | file mapping |
| Tied embedding/classifier | HBM (it is the classifier) | DDR | file mapping |
| Loader scratch | DDR, bounded | DDR | n/a |

### 4.2 Copy path

1. **Reserve one region per role per socket.** On one socket that is a weights region
   (HBM), an embedding region (DDR) and an optional packed region (C5/C6). Sizes come
   from manifest sums. Each region is bound before it is touched (A2).
2. **Parallel `pread` by the owning team.** The F2 team runs the copy, so pages are
   first touched by the socket that owns them; `MPOL_BIND` already fixes the node.
   - Each worker reads a contiguous byte range of the source tensors with `pread`
     straight into the destination, in chunks of at most 64 MiB.
   - Short reads, `EINTR` and unexpected EOF are handled (study §5.4).
   - Without F2 (`omp` backend), the coordinator thread copies alone. That is slower
     but still correct.
3. **Release the source.** Unmap the source mapping after loading, unless
   `LLAMA_PLACEMENT=file`. Page-cache residency of the file is reported but never
   evicted (study §5.3).
4. **Protect and verify.** `region_protect`, then `region_verify` (A3) before warm-up.
5. **Report the cost.** `timing_seconds.placement_populate` and
   `timing_seconds.weight_pack` are reported separately from `checkpoint_load`.

### 4.3 Q8 embedding

- **Row-wise dequantization.** `runq.c` stops expanding the full table into FP32
  (`MEM_EMBEDDING`, about 1.96 GiB on 8B, **modeled** in study §8.1). `forward`
  dequantizes the selected row on demand from `q_tokens`, which are placed per §4.1.
- **Same formula.** The row is computed with the same arithmetic as the scalar
  `dequantize` (`src/runq.c:116`), so results are bitwise equal.
- **No change for FP32.** FP32 keeps its embedding row `memcpy`
  (`src/run.c:159-160`).

### 4.4 Knobs and report

- `LLAMA_PLACEMENT=file|ddr|hbm` and `LLAMA_THP=off|on`.
- The report shows owned and packed weight bytes, every region, the verification
  results, and `mapped_checkpoint` kept separately from owned bytes.

## 5. Local tests

- **macOS, portable backend:**
  - The fixtures under `LLAMA_PLACEMENT=ddr` produce **bitwise** the same logits as
    `file`, for FP32 and INT8, shared and unshared.
  - The Q8 row-wise embedding is bitwise identical to the full-table lookup on every
    fixture token.
  - With `--json`, `regions[]` sums equal the manifest sums plus padding.
- **Short reads:** a test-only `LLAMA_LOADER_CHUNK=7` forces short chunk boundaries,
  and an interposed `pread` wrapper returns partial reads in the test build.
- **Leaks:** `make MODE=sanitize test` reports no leaks or double unmaps on any error
  path (truncated file, bind failure injection).

## 6. Cluster run

`cresco8-placement.sbatch STAGE=check PHASE=A4`. **Estimated** at 45 minutes. It needs
the user's go-ahead to submit.

| Run | Gate |
| --- | --- |
| 1B and 8B, FP32 and INT8: `compare_engines.py` `file` vs `hbm` vs `ddr` (staged file for `file`) | Bitwise identical logits at all prompt positions |
| `validate.py` with `LLAMA_PLACEMENT=hbm` | Same results as today. FP32 gated; INT8 recorded. |
| Startup breakdown for 8B FP32/INT8 | **Measured** load, populate, verify; peak RSS; page-cache footprint |
| Full verification with the 8B weights on HBM | 100% on node 2 |

## 7. Exit criteria

- Logits are bitwise equal across placements.
- Weight regions are 100% verified on the requested node.
- Startup memory peak is bounded: at most owned weights + DDR embedding + one 64 MiB
  chunk per worker, plus RSS of file pages, which are reported as page cache.
- Q8 runs no longer allocate the 1.96 GiB FP32 embedding.

## 8. Risks and open questions

- **Startup time.** FP32 8B is about 30 GiB of copying. At an **estimated** few GB/s
  of `pread` from `/dev/shm` or the parallel filesystem, startup can take tens of
  seconds. That is acceptable but must be reported. A prepacked checkpoint is future
  work (study §5.4).
- **Source location on the cluster.** The checkpoint source may be on the parallel
  filesystem rather than `/dev/shm`. Record which, since it dominates load time but
  not inference.

## 9. Not in this phase

- Two-socket shards, row slicing or column packing (B1).
- BF16 and INT8 VNNI packing (C5, C6).
- KV and workspaces (A5).
