#ifndef LLAMA_LOADER_H
#define LLAMA_LOADER_H

#include "numa.h"
#include "shards.h"

/* Owned weight loader (A4). LLAMA_PLACEMENT selects where the checkpoint's tensors live while the model runs:
 *   file  the read-only file mapping, exactly as before (default);
 *   ddr   engine-owned anonymous regions bound to the first socket's DDR node;
 *   hbm   the same, bound to its HBM node, except an untied input embedding, which goes to DDR.
 * Every view of a weight is derived from a file offset, so a WeightStore only has to answer "where does the byte at
 * this file offset live now". Owned bytes keep the file offset's alignment modulo 64, so kernels see the same
 * alignment as on the mapping. */
enum { WEIGHT_SEGMENTS = 4 };

typedef struct {
  size_t src_begin, bytes; /* the file range [src_begin, src_begin + bytes) */
  size_t dst_offset;       /* its place inside the region of its class */
  int embedding;           /* 1: embedding class (DDR under hbm), 0: weights class */
} WeightSegment;

typedef struct {
  const char *mode; /* file | ddr | hbm */
  const char *mapped;
  int nseg;
  WeightSegment seg[WEIGHT_SEGMENTS];
  NumaRegion region[2]; /* [0] weights, [1] embedding; unused ones have no mapping */
  size_t owned_bytes;
} WeightStore;

/* Pure pieces, exposed for tests. Groups the manifest's tensors into at most WEIGHT_SEGMENTS maximal file ranges of one
 * class and lays them out back to back, keeping each range's alignment modulo 64. An untied input embedding (a classifier that is
 * not an alias of it) is the only embedding-class tensor. Returns the segment count and the byte size of the weights and embedding regions. */
int weights_plan(const TensorManifest *m, WeightSegment *seg, size_t *weights_bytes, size_t *embedding_bytes);

/* LLAMA_PLACEMENT validated: "file" when unset; fails on anything but file, ddr or hbm. */
const char *placement_mode(void);
/* The mode the loaded model actually uses ("file" before any load), for the report. */
const char *placement_active(void);
/* LLAMA_HBM_BUDGET_GIB (default 56) in bytes. */
size_t placement_hbm_budget(void);

/* Fills `ws` for the already validated and mapped checkpoint. In file mode the mapping stays. In an owned mode the
 * socket-0 team preads every tensor straight into its region (or the coordinator does when there is no team), the
 * regions become read-only, their page placement is verified, and the mapping is unmapped (*mapped = MAP_FAILED). */
void weights_open(WeightStore *ws, int fd, float **mapped, size_t file_bytes, const TensorManifest *m);
/* The byte that file offset `offset` names, in whichever place it lives now. */
const char *weights_at(const WeightStore *ws, size_t offset);
void weights_close(WeightStore *ws);

#endif
