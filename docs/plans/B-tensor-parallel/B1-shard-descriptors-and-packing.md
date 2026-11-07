# B1. Shard descriptors and packing

## 1. Goal and status

Build each socket's weight shard from the validated manifest, directly into that
socket's bound region, and populate it with that socket's team.

- **Output-row partitions** are contiguous row copies.
- **Input-column partitions** (Wo, W2) are packed through bounded DDR scratch.
- **Q8 scales** travel with their values and stay aligned to their groups.

Correctness is shown by reconstructing every original tensor exactly. B1 changes no
computation.

Status: **implemented and tested locally (macOS, virtual regions); cluster `check` pending, and it inherits A4's
open cluster check**. Branch: `feature/tp-local-b` (commit `feat: add shard descriptors, validation and row/column packing`).

Implementation notes:

- **API** (`include/shards.h`, `src/shards.c`): `shards_plan_try`/`shards_plan` (validates, takes the `Config` for the
  head counts), `shards_find`, `shards_bind`, `shards_load`, `shards_free`. The proposed signatures changed:
  - `shards_load(set, fd, socket, team)` loads one socket's shards with that socket's team (`NULL` = calling thread);
    GLOBAL shards load with socket 0. It uses plain `calloc`/`malloc`, not `memory_alloc`, so the two sockets'
    coordinators may load concurrently without touching the engine's global accounting.
  - Placement is planned separately from loading: `shards_plan` lays out each socket's shards back to back (values,
    then scales, each 64-byte aligned) and reports `region_bytes[s]` and `global_bytes`; the caller reserves the
    regions and calls `shards_bind`. The `Shard` descriptor gained `alias` (tied classifier shard that also serves the
    embedding) and `q_offset`/`s_offset`.
- **Partitions** exactly as §4.1. FP32 norms are `COPY` on every socket, also in Q8 files. Every other weight is
  `ROWS` or `COLS`; the vocabulary split gives the remainder to the last socket; a tied model has no `GLOBAL` shard
  (the classifier's `ROWS` shards carry `alias`), an untied one has one `GLOBAL` embedding.
- **S = 1 is the same code**: the planner and loader take no special branch (`ROWS` shards with the full range,
  `COLS` shards with the full column range, which is a plain copy).
- **Rejection** (with messages): sockets other than 1 or 2; `n_kv_heads` or `n_heads` not splittable into complete
  GQA groups; `hidden_dim` or `dim` odd for S = 2; Q8 group size not dividing `dim/S` or `hidden/S` (never
  re-quantized); a group size different from the checkpoint's. `gs = 3` cannot reach B1 with `dim = 16`, because the
  engine's own check (`dim % gs`) comes first; the test uses `dim = 24`, `gs = 8` (`dim/S = 12`).
- **Packing.** `ROWS`/`COPY`/`GLOBAL`: one read of the values and one of the scales per shard, split by bytes among the
  team's workers (the same slicing as the owned loader, sharing `loader_read_exact`). `COLS`: blocks of complete source
  rows are read into scratch (`scratch_limit`, default 64 MiB, at least one row) and each worker packs the rows it
  read, so a block costs one barrier. `scratch_peak[s]`, `rows_seconds[s]` and `cols_seconds[s]` carry the numbers the
  report needs.
- **Tests** (`tests/shards.c`, `make test`). 96 reconstruction cases: S = 1 and 2, FP32 and Q8 with gs 2, 4, 8, shared
  and unshared, odd (37) and even (128) vocabulary, team sizes 1, 3 and 7 (and no team), plus a 7-row scratch block;
  a larger GQA case (6 heads, 2 KV heads). Every value and every scale of every tensor is rebuilt from the shards and
  compared **bitwise** with the file bytes; coverage is exactly once (twice for `COPY`). Ownership: with portable
  regions, each shard's values and scales lie inside its socket's region (or the global area), 64-byte aligned. Scratch
  never exceeds the limit. ASAN/UBSAN and TSAN clean.
- **Not done here.**
  - **A4's `weights_open` is not routed through `shards_load` yet.** Its `WeightStore` addresses weights by file
    offset, and every kernel pointer is built from that. Moving it to shard pointers changes how `run.c`/`runq.c`
    address weights, which is B5's wiring, and must be bitwise-checked against the staged baseline that A4's pending
    cluster check establishes. Until then S = 1 and S = 2 share `shards_plan`/`shards_load`, and the engine's default
    path is unchanged. **Open risk: B1 builds on A4, whose cluster check is still pending.**
  - The format record in the report (packing version `SHARD_PACKING_VERSION`, partition, source identity, shard
    shapes) is not written yet; B5 adds it with the other `LLAMA_SOCKETS=2` report fields.
  - Cluster (`cresco8-tp.sbatch PHASE=B1`, dry-run verified): `test-shards verify FILE fp32|int8 hbm 56` is the test-only
    mode of §6. It shards a real checkpoint over both sockets (the machine's topology, teams pinned, regions bound),
    compares every shard with the file bitwise, verifies all pages of each region and prints the row-copy and
    column-pack times and the scratch peak as JSON; it also runs on the virtual topology (checked here on the `tp`
    fixtures, FP32 and Q8, shared, unshared and odd vocabulary). Still to be run on 1B and 8B.

## 2. Depends on / unblocks

- Depends on: F1 (manifest) and A4 (owned loader, regions, parallel `pread`).
- Unblocks: B2. It also defines the descriptors that C5, C6 and D1 pack AMX layouts
  from.

## 3. Reuse

| Code | Location | Use |
| --- | --- | --- |
| Manifest | F1 `TensorDesc` | Rows, columns, offsets and scale offsets |
| Loader | A4 | Region per {socket, role}; parallel `pread`; owner-team population; verification |
| Q8 layout | `src/runq.c:136-149` | Each tensor stores `q[rows·cols]`, then `s[rows·cols/gs]`. Row r's scales are `s + r·(cols/gs)` (`src/kernels_simd.c:506-508`). |
| Group checks | `src/runtime.c:100-120` | `gs` divides `dim` and `hidden_dim` |

## 4. Changes

### 4.1 Partition table

Generic form of study §6.1. `S` is the socket count, 1 or 2:

| Tensor | Full `[out][in]` | Socket s shard | Kind |
| --- | --- | --- | --- |
| Wq | `dim × dim` | rows `[s·dim/S, (s+1)·dim/S)` | `ROWS` |
| Wk, Wv | `kv_dim × dim` | rows `[s·kv_dim/S, …)` | `ROWS` |
| Wo | `dim × dim` | columns `[s·dim/S, …)`, all rows | `COLS` (packed) |
| W1, W3 | `hidden × dim` | rows `[s·hidden/S, …)` | `ROWS` |
| W2 | `dim × hidden` | columns `[s·hidden/S, …)` | `COLS` (packed) |
| Classifier | `vocab × dim` | rows `[s·vocab/S, …)` | `ROWS` |
| Norm weights | `dim` | full copy per socket | `COPY` |
| Untied embedding | `vocab × dim` | one copy in the coordinator socket's DDR | `GLOBAL` |
| Tied embedding/classifier | `vocab × dim` | vocabulary-row shards, owned per socket; the coordinator fetches an embedding row from its owner | `ROWS` + `alias` |

### 4.2 Validation, in addition to the F1 checks

- `n_kv_heads % S == 0`, so complete KV heads are owned.
- `n_heads / n_kv_heads` (GQA) groups stay complete. The query-head ranges must equal
  `kv_mul ×` the KV-head ranges.
- `hidden % S == 0`.
- `vocab` may be odd. For example, 128256 / 2 = 64128 is even, but the general case
  gives the remainder to the last socket.
- **Q8.** `(dim/S) % gs == 0` and `(hidden/S) % gs == 0`, i.e. `2048 % gs` and
  `7168 % gs` for 8B. With `gs = 64`, both hold.
- **Incompatible group sizes are rejected** with a clear error. Never re-quantize
  (study §5.4).

### 4.3 Descriptor

```c
/* Proposed, include/shards.h. */
typedef enum { SHARD_ROWS, SHARD_COLS, SHARD_COPY, SHARD_GLOBAL } ShardKind;
typedef struct {
  const TensorDesc *src; ShardKind kind; int socket;
  int row0, rows, col0, cols;         /* in the source tensor */
  void *q; float *s;                  /* destination values / Q8 scales in the socket region */
  size_t ld;                          /* destination row stride in elements (= cols) */
} Shard;
typedef struct { int sockets; Shard *t; int count; } ShardSet;

void shards_plan(ShardSet *set, const TensorManifest *m, int sockets, int gs);   /* validates */
void shards_load(ShardSet *set, int fd, Team *team);                             /* per-socket team */
```

The one-socket case is just `S = 1`: every tensor is `ROWS` or `COPY` with the full
range. A4's loader becomes `shards_load` with `S = 1`, so tracks A, B and C share a
single code path.

### 4.4 Packing

**`ROWS`.** One contiguous byte range per tensor per layer: values, then the
corresponding contiguous scale range. It is read with `pread` directly into the
destination.

**`COLS`.** Packed through bounded DDR scratch:
1. Read blocks of complete source rows (at most 64 MiB) into scratch in the owner
   socket's DDR.
2. Copy each row's `[col0, col0+cols)` slice into the destination row, with stride
   `cols`.

For Q8, copy the matching scale slice `[col0/gs, (col0+cols)/gs)` of each row too. The
copy is batched; there is never one `pread` per slice (study §5.4).

**Who loads each shard.** The socket team that owns the shard writes it, and every
destination region is `MPOL_BIND` to the owner's HBM node.

**Format record.** The report records the packing version, partition, source identity
(checkpoint SHA, or size plus mtime) and shard shapes.

## 5. Local tests

These run on macOS, on the `tp` fixture and on random tensors.

- **Reconstruction.** For S = 1 and S = 2, for FP32 and Q8 (gs 2, 4, 8), and for shared
  and unshared models, rebuild every full tensor from its shards. Every value and scale
  must be **bitwise** equal to the source.
- **Rejection.**
  - `gs = 3` with `dim/S = 8` is rejected.
  - `n_kv_heads = 1` with `S = 2` is rejected (the existing fixture).
- **Bounded scratch.** With a test-only 7-row block size, scratch never exceeds the
  limit.
- **Ownership.** With the virtual topology, each shard's destination belongs to the
  region of its socket.

## 6. Cluster run

`cresco8-tp.sbatch STAGE=check PHASE=B1`. **Estimated** at 45 minutes. It needs the
user's go-ahead to submit.

| Run | Gate |
| --- | --- |
| Shard 8B FP32 and INT8 with S = 2, teams pinned to sockets 0 and 1, HBM placement | Full reconstruction check (a test-only mode compares against the source file) |
| Verification | Socket 0 shards 100% on node 2; socket 1 shards 100% on node 3 |
| Startup breakdown | **Measured** row-copy and column-pack time per socket, and the scratch peak |

## 7. Exit criteria

- Reconstruction is bitwise exact for every tensor and scale.
- Per-socket placement is verified.
- Packing time and scratch peak are reported.
- Invalid partitions are rejected.

## 8. Risks and open questions

- **Column packing cost.** Wo and W2 hold about 1/3 of the layer bytes (**modeled**:
  `dim² + dim·hidden` out of the layer total). Packing them reads full rows through
  DDR. Expect the startup cost to grow by roughly the read time of those tensors.
  Measure it before considering a prepacked format.
- **Odd vocabulary split.** The classifier's vocabulary split with a remainder makes
  the logit gather buffer offsets non-uniform. Test it with an odd fixture vocabulary.

## 9. Not in this phase

- Computation on shards (B2–B5).
- AMX/BF16 packed layouts. C5 and C6 add pack hooks that consume these same `Shard`
  descriptors.
- Prepacked checkpoints.
