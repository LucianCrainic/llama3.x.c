# HBM/DDR planned placement (merged with develop, 24 November 2026)

`LLAMA_PLACEMENT=hbm-only|ddr|hybrid` plans every large allocation of a run before any of it exists, admits the plan
against per-socket, per-tier budgets, reserves one bound arena per (socket, role, tier), and verifies the pages of each
arena afterwards. The branch `feature/hbm-ddr-placement` added it at 3098383; this document describes the merge into
`develop` b9b9bbb, which had meanwhile added memory users the planner did not know. Results and job ids are in
`docs/hbm-ddr-placement-validation.md`. The knobs are in `docs/configuration.md` (Planned placement). Numbers here are
labelled measured, derived or modelled.

## 1. What the planner owns

Per socket (`tp.c`, `placement_layout`), each an arena with a role, a tier (HBM or DDR node of the socket), a requested size
and the size rounded to its page size (4 KiB, or 2 MiB with `LLAMA_THP=on`):

| arena | role | tier |
|---|---|---|
| weight shards of the first `LLAMA_HYBRID_WEIGHTS_HBM_LAYERS` layers / the rest | `weights` | HBM / DDR |
| K and V caches of the first `LLAMA_HYBRID_KV_HBM_LAYERS` layers / the rest | `kv.k`, `kv.v` | HBM / DDR |
| persistent BF16 copy of K and V (`LLAMA_ATTN=amx`), the same layer split | `kv.bf16` | HBM / DDR |
| activations, attention workspace, BF16 panel | `work` | hot tier |
| two-socket exchange partials and flags | `exchange` | hot tier |
| logits (and all-position logits with `LLAMA_PREFILL_LOGITS=all`) | `logits` | hot tier |
| packed prefill copies, packed classifier | `packed` | `LLAMA_HYBRID_PACKED` |
| untied embedding | `embedding` | cold tier |
| loader scratch (released after the load) | `scratch` | cold tier |

The charge of a node is the sum of the rounded arenas on it (derived; `tests/test_tp.py` checks the identity). The
over-budget fatal error is unchanged and remains the default for `hbm-only`; spilling happens only under an explicit
`hybrid`. Engine defaults of the default placement are untouched.

## 2. The develop additions and how each is planned

1. **Persistent BF16 K/V copy** (`k_attention_set_kv_bf16`, 5debcf4). On the default placement it is carved from the
   work region, as on develop. Under a plan it has its own arenas, one copy per layer of
   `k_attention_kv_bf16_bytes(kv heads per socket, head_dim, seq_len, 1)` bytes, in the tier of the layer's caches, so a
   hybrid split keeps a layer's copy beside its K and V. `k_attention_set_kv_bf16_layers` binds each layer's copy and key
   cache pointer up front (a layer's slot no longer depends on the order the layers first run in). The copy is charged to
   both budgets, zeroed by the socket's team at setup (first touch on its node, like the caches), verified over its
   active prefix (`region_verify_prefixes`, one span per K/V head, `ceil(position / 32)` key blocks), and reported as
   `placement_plan.kv_bf16_layer_bytes`, the `kv.bf16` arenas and regions, and `kv.bf16_copy` (reserved, active,
   resident bytes). `tools/check_placement.py --planned` requires the report and the verified regions.
2. **Per-channel INT8 (format 4, gs 0).** The work size divided by the group size; a per-channel file has one scale per row
   and input, so the planner now counts two scales per padded row. Shard sizes were already right (`group_size == cols`).
3. **gs32 and gs64 AMX-INT8, VNNI GEMV.** Packed sizes come from the same `pack_bytes_of` as the legacy path (tiles, then
   with the tiled kernel the transposed scales per `tp_group`), so `LLAMA_HYBRID_PACKED=hbm|ddr` sizes and places gs32,
   gs64 and per-channel copies alike.
4. **Fused RMSNorm + tile-order quantizer.** The merge keeps `norm_quantize_batch` in the batched prefill loop and the
   branch's `kv_layer()` accessor.
5. **KV indexing.** Every access goes through `kv_layer(sk, l)`. The two per-socket pointers `key_cache`/`value_cache` are
   gone: the legacy placement builds the same per-layer table the plan builds, and the prefault of the KV pages iterates
   the arenas of a plan (both tiers and the BF16 copy) or the two legacy regions. The only layer-to-offset arithmetic is
   the table's construction (checked by `test_no_raw_kv_indexing_outside_the_kv_table`).
6. **Interleaved Wo/W2 reduction, exchange, team prefault.** The exchange arena is sized by `exchange_region_bytes` (3 x
   rows x dim floats, partials and flags); a planned two-socket run now applies the same block-flag capacity check as
   `tp_create`. The team prefault of the KV pages is done under a plan too (it used the legacy pointers and would have
   written through NULL). The `tp->plan.mode` guards of `tp.c` are enumerated in `test_planned_split_kv_prefault_and_exchange`;
   each is either a legacy-only step whose plan equivalent exists (work memset: `CMD_SETUP`; KV table; BF16 carve; free) or an
   admission check.
7. **`scripts/hbm-ddr.sbatch`.** It no longer forces `LLAMA_KERNELS=avx512`, `LLAMA_PREFILL_BATCH=64`, tiled attention,
   split decode, the head layout or `LLAMA_THP=off`: engine defaults apply unless `KERNELS`, `PREFILL_BATCH`,
   `PREFILL_LOGITS`, `KV_LAYOUT`, `ATTN`, `ATTN_DECODE`, `THP` are set; `LEGACY=1` restores the old values;
   `DRY_RUN=1` prints the environment.

## 3. Merge resolutions

- `src/attention.c`: `k_attention_bytes_rows_for` (branch: the worker count of the planned team) now carries develop's
  AMX BF16 workspace; `k_attention_bytes_rows` calls it with `max_threads()`.
- `src/tp.c`: the prefill loop takes develop's `norm_quantize_batch` and the branch's `kv_layer()`; the work size keeps
  develop's per-channel rule and its BF16 term (legacy only, see above); the two-socket cache assignment is gone
  (item 5).
- `tests/test_tp.py`: both sides' tests are kept.

## 4. What is verified

Local (`make test`, virtual topologies, software AMX tile model): `tests/test_tp.py` new tests, one per hazard; see the
validation document. On hardware: `make test`, the ThreadSanitizer tp and Q8 tests, the bitwise comparison with develop
on the default path, planned modes with `check_placement.py --planned` on every report. Not verified: the page checks of
the BF16 copy on a virtual topology (simulated plans skip page queries); that is covered by the hardware runs only.
