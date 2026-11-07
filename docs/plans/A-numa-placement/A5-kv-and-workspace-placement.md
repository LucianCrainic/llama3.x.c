# A5. KV cache and workspace placement

## 1. Goal and status

Put the KV caches, activation vectors, attention workspace and score buffer in the
owning socket's bound regions, first touched by the owner team. KV capacity is
reported as reserved, resident and active bytes. Inference math is unchanged, and
logits must be bitwise equal to A4.

Status: **implemented; cluster check passed** (see Cluster results).

Implementation notes:

- `RunMemory` (`src/loader.c`): under `file` every buffer is the same `memory_alloc` as before. Under `ddr`/`hbm` the K and
  V caches are two regions (`kv.k`, `kv.v`) whose pages the first write at each position creates, and every other
  run-state buffer (activations, scores, logits, RoPE tables, the split-K attention workspace, the Q8 scratch) is carved
  from one `work` region by a sizing pass followed by the real pass. The owning team writes the work region once.
  Accounting follows the extent: `kv_cache_allocated_peak` is the reservation, residency comes from the query.
- **Reporting.** `kv = {reserved_bytes, resident_bytes, active_bytes, node, verified, full, pages_queried,
  pages_wrong_node, pages_missing, after_prompt: {...}}`. `resident_bytes` is `null` unless a `move_pages` query ran
  (`LLAMA_KV_VERIFY=0` skips it). The snapshot after the prompt and the final one run between timed regions: their
  duration is removed from TTFT and `total_inference` (`verify_excluded_s`) and added to `placement_verify`.
- A KV page on the wrong node is fatal under strict placement, like a weights page.
- `tools/check_placement.py` judges one report (nodes, full verification, nothing elsewhere or missing, and
  active ≤ resident ≤ reserved plus a page per layer) and is what the cluster phase uses.
- **Local evidence.** Accounting formulas hold for `file` and owned placements on both engines, a run to the exact
  context limit, bitwise logits (A4 tests cover the new regions too), sanitizers. **Not yet run:** residency
  verification of real KV regions, which needs Linux.

## 2. Depends on / unblocks

- Depends on: A4 and F2 (owner team for first touch).
- Unblocks: A6 and B3. B3 splits these per socket and per local head.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| FP32 `malloc_run_state` | `src/run.c:59` | KV: `L·seq_len·kv_dim` floats each. `att`: `n_heads·seq_len`. The rest are vectors. |
| Q8 `malloc_run_state` | `src/runq.c:73` | Same, plus `xq`/`hq` and separate `k`/`v` |
| FP32 direct KV write | `src/run.c:171-177` | The QKV kernel writes K/V straight into the cache row, so first touch follows the computing thread |
| Q8 KV copy | `src/runq.c:254-258` | `memcpy` on the master thread |
| `k_attention_init` | `src/kernels_simd.c:675` | `partial`: `n_heads·max_chunks·(head_dim+2)` floats |
| Context bound | `src/run.c:211` (`seq_len = context ? context : min(seq_len, 4096)`) | Determines KV reservation |

## 4. Changes

### 4.1 Where each buffer goes

- **KV caches** become two regions per socket (`ROLE_KV_K`, `ROLE_KV_V`) on the
  selected node, reserved for `L·seq_len·kv_dim·4` bytes each. Pages are populated by
  first write at each new position, not up front. That keeps "reserved" and "resident"
  distinct.
- **Activations, the attention workspace, the score buffer and the logits** go into a
  single `ROLE_WORK` region on the socket's node, populated at startup by the owner
  team, which writes zeros once.

### 4.2 First touch under each backend

With `teams`, the K/V rows written by QKV are touched by socket-local workers. That is
already correct under `MPOL_BIND`, which wins regardless of the touching thread.

With the `omp` backend and no team, the bind policy alone guarantees the node.

### 4.3 Reporting KV

The report's `kv` object (F3) separates three quantities:

| Field | Definition |
| --- | --- |
| `reserved_bytes` | The virtual extent |
| `resident_bytes` | From `region_verify` on the KV regions after the prompt and at the end |
| `active_bytes` | `2·L·position·kv_dim·4` |

### 4.4 Verification timing

A KV verification pass runs after prefill and at completion. It is never timed inside
decode (A3).

## 5. Local tests

- **Bitwise logits:** the fixtures under `LLAMA_PLACEMENT=ddr` match `file` bitwise.
- **KV accounting:**
  - After N tokens, `active_bytes` equals the formula.
  - On Linux, `resident_bytes` is at least `active_bytes` and at most `reserved_bytes`
    plus a page per layer.
- **Context limit and reset:** a run up to `--context-length` leaves no access beyond
  the reservation (ASan build), and the chat-mode multi-turn tests in
  `test_runtime.py` still pass.

## 6. Cluster run

`cresco8-placement.sbatch STAGE=check PHASE=A5`. **Estimated** at 30 minutes. It needs
the user's go-ahead to submit.

| Run | Gate |
| --- | --- |
| 8B FP32/INT8, `--context-length 32768`, a 2048-token prompt, 128 generated tokens, HBM | KV pages 100% on node 2 after the prompt and at the end; `active` ≤ `resident` ≤ `reserved` |
| `compare_engines.py` against A4 | Bitwise identical logits |
| Same with `LLAMA_THP=on` | THP backing reported for weights and KV |

### Cluster results (2026-11-07)

cresco8-hbm14 (Flat mode, 2 × Xeon Max 9480), gcc 11.4.1, `OPENMP=1`, commit `fec5f44`, `STAGE=check`, `HF_GATE=0`. Job 5693937, same run as A4.

- **Residency.** For 1B and 8B, FP32 and Q8, with THP off and on: the K, V and work regions are reserved on the requested
  node, and the KV pages that exist are verified on it with none elsewhere (`elsewhere=0`). `active <= resident <=
  reserved` holds after the prompt and at the end (8B FP32, 2048-token prompt, 32768 context: active 579 MB, resident
  671 MB, reserved 8.59 GB; the unwritten remainder shows as missing pages, as designed).
- **Logits.** `hbm` against `file` stays bitwise identical with microbatched prefill (M = 32) on all four model and
  precision combinations.
- **Not measured.** The decode effect of socket-local KV at long context belongs to A6.

## 7. Exit criteria

- KV and workspaces are verified on the requested node.
- KV reserved, resident and active bytes are reported.
- Logits are bitwise unchanged.

## 8. Risks and open questions

- **Capacity at long context.** On the FP32 path, KV for one 128K request is
  **modeled** at 32 GiB total (study §8.2). On one socket, 13.98 GiB of weights plus
  32 GiB of KV is within 64 GiB nominal HBM, but A6 does not need that size. The
  per-socket budget check (A2) must fail early if `reserved` exceeds the configured
  budget (`LLAMA_HBM_BUDGET_GIB`, default 56; study §8.2).

## 9. Not in this phase

- KV split by heads across sockets (B3).
- Block-allocated or tiered KV (out of scope for all phases here).
- BF16 KV.
