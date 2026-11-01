# C3. Batched causal attention and KV append

## 1. Goal and status

Process M query rows of a microbatch together:

1. Append all M K/V rows to the cache, after RoPE at positions `pos0..pos0+M-1`.
2. Run causal grouped attention, where query row i attends to positions
   `0..pos0+i`.
3. Reuse the numerically stable chunk merge.

The result must match per-row attention within the existing budget.

Status: **planned**. Branch: `feature/prefill-c3-causal-attention`.

## 2. Depends on / unblocks

- Depends on: C2. It also needs F2 (team task queue), and B3's local-head parameters
  where available.
- Unblocks: C5, D1.

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| `k_attention` and chunk kernel | `src/kernels_simd.c:735`, `:288` | (group, chunk) tasks; scores in `att[h·seq_len+t]` |
| Merge | `src/kernels_simd.c:760-780` | Combines chunks by max and weighted sum |
| `k_rope_table` / `k_rope_apply` | `include/kernels.h:21-22` | One table per position: M tables per microbatch |
| B3 local counts and strides | B3 | Used so that D1 needs no second attention implementation |

## 4. Changes

### 4.1 Signature

```c
/* Proposed. */
void k_attention_batch(float *out /* M x n_heads_local*head_dim */, const float *q /* M x ... */,
                       const float *kc, const float *vc, int kv_stride,
                       int n_heads_local, int n_kv_heads_local, int head_dim,
                       int seq_len, int pos0, int m, AttentionWork *work);
```

### 4.2 Algorithm

1. **RoPE.** Compute M RoPE tables (`2·head_dim` floats each) and apply them per row.
2. **KV append.** Write K/V rows `pos0..pos0+M-1` into the cache, using the local
   layout from B3 when S = 2. Every row is written before any attention task starts,
   with a team barrier in between.
3. **Tasks.** One task per `(kv group g, query block of size b_q, position chunk c)`,
   with `b_q` around 4–8 query rows sharing one K/V chunk load. GQA multiplies the reuse
   by `kv_mul = 4` on 8B.
   - For query row i, positions beyond `pos0+i` are masked: chunks entirely beyond the
     row's limit are skipped, and the boundary chunk is masked partially.
   - Each `(row, head, chunk)` gets its own partial slot `[acc, m, l]`.
4. **Merge.** Merge per (row, head) over chunks with the existing formula. The merge is
   parallel over `M × n_heads_local`.

### 4.3 Workspace

`partial` grows to `M_max · n_heads_local · max_chunks · (head_dim + 2)` floats
(**modeled** for 8B with 128 chunks at M = 128: 128 × 32 × 128 × 130 × 4 B ≈ 273 MB
on one socket, and half that per socket when S = 2). For
long prompts, either:

- cap `max_chunks` for batched calls; or
- process query blocks in waves so the workspace stays bounded.

The plan picks the wave approach and reports the workspace bytes.

### 4.4 Scores

The score buffer becomes per task, held in thread-local stack or workspace, instead of
`att[h·seq_len+t]`, because M rows would otherwise multiply it by M.

## 5. Local tests

These run on macOS (scalar) and on AVX-512 hosts.

- **Matches per-row attention.** Batched causal attention for M ∈ {1, 2, 7, 32},
  `pos0` ∈ {0, 31, 250, 4000}, at head_dim 64 and 128, with GQA 1 and 4, matches
  per-row `k_attention` at each row's position within `2e-5` absolute.
- **Causality.** Changing K/V at positions `> pos0+i` must leave row i unchanged,
  bitwise. This is checked by perturbation.
- **Context limit.** A microbatch that reaches the context limit exactly; `pos0 + M`
  crossing the 32- and 256-position chunk boundaries.
- **Fixture forward.** C2 GEMM plus C3 batched attention vs the C1 reference, within
  the fixture tolerance at all positions.

## 6. Cluster run

`cresco8-prefill.sbatch STAGE=check PHASE=C3`. **Estimated** at 45 minutes. It needs
the user's go-ahead to submit.

| Run | Gate or record |
| --- | --- |
| `test-kernels` attention cases, 56 workers | Pass |
| `compare_engines.py` 8B FP32: M = 64 (GEMM + batched attention) vs per-token, prompt lengths 86 and 2048 | max\|Δlogit\| ≤ 1e-3, argmax 100% |
| Attention share of prefill time at prompts 2048 and 8192, M = 64 | **Measured**. Long prompts make attention quadratic; record when it overtakes GEMM. |

## 7. Exit criteria

- Batched causal attention matches per-row attention within budget, and causality is
  exact.
- Real-model logits agree within 1e-3.
- The workspace is bounded and reported.

## 8. Risks and open questions

- **Attention dominates long prompts.** For prompts of 8K and more, causal attention is
  O(P²) and is not helped by AMX in this plan. Record its share. AMX attention (QKᵀ and
  PV as tile products) is a possible later phase, not planned here.

## 9. Not in this phase

- AMX (C4–C6).
- Flash-style fused attention.
- Two sockets (D1).
