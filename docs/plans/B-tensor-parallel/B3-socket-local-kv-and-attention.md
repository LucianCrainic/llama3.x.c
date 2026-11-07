# B3. Socket-local KV cache and attention

## 1. Goal and status

Each socket stores K/V **only for its own KV heads**, in its own region. It runs
grouped attention for its own query heads on its own team, with local head indexing
and per-team task queues. The KV history, the attention scores and the per-head outputs
never cross sockets (study §7).

Status: **planned**. Branch: `feature/tp-b3-local-kv`.

## 2. Depends on / unblocks

- Depends on: B2 (local QKV output), A5 (KV regions) and F2 (team-aware attention).
- Unblocks: B5, and C3/D1 (batched causal attention on local heads).

## 3. Reuse

| Code | Location | Note |
| --- | --- | --- |
| `k_attention` | `src/kernels_simd.c:735` | GQA: `h0 = g·kv_mul`. Cache layout `[pos][kv_head][head_dim]`, stride `kv_dim`. |
| `attention_chunk_avx512` | `src/kernels_simd.c:288` | Per (group, chunk). Writes `att[h·seq_len + t]`. |
| Chunk merge | `src/kernels_simd.c:760-780` | Stable max/sum merge; reused unchanged |
| `attention_chunks` | `src/kernels_simd.c:662` | After F2 it takes the team size |
| KV regions | A5 | Reserved per socket |
| `k_rope_apply` | `include/kernels.h:22` | Takes `n_heads` and `n_kv_heads` counts. Called with local counts, it rotates only the local heads, using the same table. |

## 4. Changes

- **KV layout.** Per socket s: `K_s[layer][capacity][kv_heads/S][head_dim]`, and
  likewise for V, with `kv_dim_local = kv_dim/S` (512 on 8B). Each socket has separate
  `ROLE_KV_K` and `ROLE_KV_V` regions on its HBM node. Nothing is interleaved within a
  token row (study §7).
- **Attention signature.** `k_attention` gains explicit local counts and strides:
  `n_heads_local`, `n_kv_heads_local` and `kv_stride = kv_dim_local`.
  - All indices inside the kernel are local.
  - The global head id appears only in shard metadata (B1).
  - A global head id used as a local index is the bug this phase guards against.
- **Per-team state.** Each team has its own `AttentionWork` (`partial`, `max_chunks`)
  and score buffer `att` (`n_heads_local · seq_len`), allocated in its `ROLE_WORK`
  region.
- **Task queue.** A team-local atomic counter over `(local kv group, chunk)`. With
  four local KV heads, position chunks provide the remaining parallelism.
  `attention_chunks` takes the team size, so the chunk count adapts. At short contexts
  there may be too little work for 56 workers; measure this rather than assuming it
  (study §7).
- **Positions.** The position and RoPE table are identical on both sockets, and come
  from coordinator state that is broadcast at each step.

## 5. Local tests

These run on macOS with the virtual topology `2x3`.

- **Split equals full.** The two sockets' attention outputs, concatenated by head, must
  equal one full-model `k_attention` on the same Q/K/V, within `2e-5` absolute (the
  existing attention budget, `tests/kernels.c:233-266`). Run it at positions 0, 31, 32,
  255, 256, 1000 and 4095, and with team sizes 1, 3 and 7.
- **Local indexing check.** Socket 1 data placed at global offsets in a canary buffer
  is never read. An ASan build plus a guard region after the local cache checks this.
- **Lifecycle.** Cache reset, the context limit, and chat multi-turn continuation on
  the `tp` fixture.

## 6. Cluster run

`cresco8-tp.sbatch STAGE=check PHASE=B3`. **Estimated** at 15 minutes. It needs the
user's go-ahead to submit.

- Run the attention kernel tests with 2 × 56 pinned workers.
- Measure attention task time per socket at contexts 512, 8192 and 32768 with
  synthetic Q/K/V, to check the chunk count against the team size.

## 7. Exit criteria

- The local-head attention concatenation matches the full attention within budget.
- No out-of-range local indexing.
- Per-team chunk counts are reported.

## 8. Risks and open questions

- **Imbalance at long contexts.** When decode time is attention-dominated, both
  sockets have equal head counts, so they are balanced by construction. Any imbalance
  then comes from the memory tier or clocks. B7's per-phase breakdown shows it.

## 9. Not in this phase

- Batched/causal multi-query attention (C3, D1).
- Tiered or blocked KV.
- BF16 KV.
