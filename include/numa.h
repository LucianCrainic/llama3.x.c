#ifndef LLAMA_NUMA_H
#define LLAMA_NUMA_H

#include <stddef.h>
#include <stdint.h>

/* Socket/NUMA topology discovered in C from sysfs and the job's allowed sets, with
 * no numactl or lscpu. A node is never classified as HBM merely because it is
 * memory-only: that needs numaperf data or an explicit map. */
typedef enum {
  CLASS_NUMAPERF, /* HBM nodes identified from /sys/.../access0/initiators bandwidth */
  CLASS_EXPLICIT, /* LLAMA_TOPOLOGY=explicit:C0/H0/D0,C1/H1/D1 */
  CLASS_VIRTUAL,  /* LLAMA_TOPOLOGY=virtual:SxT: fake unpinned sockets, no nodes */
  CLASS_DDR_ONLY  /* no CPU-less memory node exists, so there is nothing to classify */
} TopologyClass;

typedef struct {
  int package; /* physical_package_id; -1 for virtual sockets */
  int ncpus;
  int *cpus;                    /* allowed physical cores, one CPU per core, ascending; NULL when virtual */
  int ddr_node, hbm_node;       /* -1 when absent */
  int ddr_allowed, hbm_allowed; /* node is in this job's allowed memory set (1 when virtual or absent) */
  size_t ddr_bytes, hbm_bytes;  /* node MemTotal, informational */
} SocketTopo;

typedef struct {
  int nsockets; /* sockets that own at least one allowed CPU */
  SocketTopo socket[2];
  int max_node; /* highest possible node id, for mbind masks; -1 when virtual */
  size_t mems_words;
  unsigned long *mems_allowed;      /* bitmask of allowed nodes, mems_words words sized from the possible nodes */
  char mems_allowed_list[128];      /* the same set as a list, e.g. "0-3" */
  char proc_mems_allowed_list[128]; /* /proc/self/status Mems_allowed_list, diagnostic only (numa_topology) */
  TopologyClass classification;
} Topology;

/* Inputs of one discovery. NULL fields mean: real /sys, `auto`, every online CPU,
 * every online node. Tests substitute fixtures here. */
typedef struct {
  const char *sysfs_root;   /* replaces "/sys" */
  const char *spec;         /* LLAMA_TOPOLOGY: auto | virtual:SxT | explicit:C0/H0/D0,C1/H1/D1 (C = package id) */
  const char *allowed_cpus; /* cpu list, e.g. "0-55" */
  const char *allowed_mems; /* node list, e.g. "0,2" */
} NumaInputs;

/* Returns 0, or -1 with a message in `err`. */
int numa_discover(Topology *t, const NumaInputs *in, char *err, size_t err_size);
void numa_topology_free(Topology *t);
/* Whether `sockets` (1 or 2) sockets can run with `placement` (file|ddr|hbm; NULL = file). */
int numa_check_placement(const Topology *t, int sockets, const char *placement, char *err, size_t err_size);
const char *numa_class_name(TopologyClass c);

/* Discovered once, before any pinning, from LLAMA_TOPOLOGY and the job's real affinity
 * and allowed memory nodes; fails the process on error. Call before threads are created. */
const Topology *numa_topology(void);
/* Same discovery and cache without failing the process: 0 and the topology, or -1 and the message. The result,
 * including a failure, is cached by the first call. */
int numa_topology_try(const Topology **topology, char *err, size_t err_size);

/* Linux affinity and NUMA online read shared with the report; -1 when unavailable. */
int numa_affinity_cpus(int *cpus, int capacity);
int numa_online_nodes(char *buf, size_t size);

/* ---- Bound memory regions (A2) --------------------------------------------------------------------------------
 * A region is one anonymous mapping placed on exactly one node before any page is touched. The owning team
 * populates it, the coordinator accounts for it, and it is released exactly once; the registry of live regions
 * releases whatever is left at exit, so fail() paths and partial startups leak nothing. */
typedef enum { BACKEND_PORTABLE, BACKEND_LINUX_BOUND } RegionBackend;
typedef enum { ROLE_WEIGHTS, ROLE_PACKED, ROLE_EMBEDDING, ROLE_KV_K, ROLE_KV_V, ROLE_WORK, ROLE_SCRATCH } RegionRole;

typedef struct {
  void *mapping_base;
  size_t mapping_bytes; /* what munmap receives; includes alignment slack */
  void *data;
  size_t logical_bytes; /* requested size */
  size_t data_bytes;    /* logical size rounded to the page (or THP) size; the extent that is bound */
  int socket, node;     /* node -1 = portable backend, not bound */
  RegionRole role;
  RegionBackend backend;
  int thp_requested, read_only, populated;
} NumaRegion;

/* Reserve and bind. A bind failure never falls back to another node. `_try` returns -1 with a message; the plain
 * form calls fail(). node < 0 selects the portable backend (plain mmap, no binding), the only one off Linux. */
int region_reserve_try(NumaRegion *r, size_t bytes, int socket, int node, RegionRole role, int thp, char *err,
                       size_t err_size);
void region_reserve(NumaRegion *r, size_t bytes, int socket, int node, RegionRole role, int thp);
void region_mark_populated(NumaRegion *r); /* coordinator accounting after the owner filled the region */
void region_protect(NumaRegion *r);        /* PROT_READ, for weights */
void region_release(NumaRegion *r);        /* idempotent; zeroes the descriptor */
void region_release_all(void);             /* also registered with atexit on the first reservation */
int region_live_count(void);
const NumaRegion *region_live_at(int index);
/* ---- Placement verification (A3) -------------------------------------------------------------------------------
 * Where do the populated pages actually live? move_pages(2) queried without migrating, in batches. */
typedef struct {
  size_t queried, on_node, wrong_node, missing, errors;
  int full;           /* every populated page was queried */
  const char *scheme; /* "full" or "stride+extent-endpoints" */
} PageCheck;

/* max_samples == 0 queries every page; otherwise a region with more pages than max_samples is sampled (first and
 * last page of every 2 MiB extent, then a deterministic stride) and is never reported as full. Returns -1 when the
 * region has no requested node or the query is unavailable (non-Linux, or move_pages blocked): then use
 * region_numa_maps and label the run unverified. */
int region_verify(const NumaRegion *r, size_t max_samples, PageCheck *out);
int region_thp_bytes(const NumaRegion *r, size_t *bytes); /* AnonHugePages overlapping the region, from smaps */
int region_numa_maps(const NumaRegion *r, char *policy, size_t policy_size, size_t *node_pages, int nodes);
size_t region_verify_samples_default(void);  /* LLAMA_VERIFY_SAMPLES, 0 = full */
int placement_strict(const char *placement); /* LLAMA_PLACEMENT_STRICT, default on for hbm and ddr */

/* Pure pieces, exposed for tests. */
void page_check_classify(const int *status, size_t n, int node, PageCheck *out);
/* Sorted page indices to query, at most max_samples of them; `extent_pages` pages per 2 MiB extent. */
size_t page_sample_plan(size_t pages, size_t extent_pages, size_t max_samples, size_t *out);
size_t smaps_thp_bytes(const char *text, uintptr_t lo, uintptr_t hi);
int numa_maps_lookup(const char *text, uintptr_t addr, char *policy, size_t policy_size, size_t *node_pages, int nodes);

/* Every region the run reserved, released or not, for the report; verification fills in the evidence. */
typedef struct {
  const void *id; /* mapping base of the reservation */
  RegionRole role;
  int socket, node, thp_requested;
  size_t bytes;
  int checked, full; /* a verification was recorded; it was full rather than sampled */
  size_t thp_bytes, pages_queried, pages_mismatched, pages_missing;
} RegionRecord;
int region_history_count(void);
const RegionRecord *region_history_at(int index);
void region_record_verification(const NumaRegion *r, const PageCheck *check, size_t thp_bytes);

int region_round(size_t bytes, size_t align, size_t *rounded); /* -1 on zero or overflow */
int region_thp_default(void);                                  /* LLAMA_THP=on */
const char *region_role_name(RegionRole role);

#endif
