/* Bound-region tests and probe.
 *   test-regions                    unit tests (portable backend everywhere; bound backend notes on Linux)
 *   test-regions fail3              reserve three regions, then fail(); prints what the exit path left behind
 *   test-regions bind NODE BYTES    reserve a region bound to NODE; exit 1 with the message on failure
 *   test-regions probe ...          see probe() */
#include "numa.h"
#include "runtime.h"
#include "workers.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void unit_rounding(void) {
  size_t out = 0;
  assert(region_round(1, 4096, &out) == 0 && out == 4096);
  assert(region_round(4096, 4096, &out) == 0 && out == 4096);
  assert(region_round(4097, 4096, &out) == 0 && out == 8192);
  assert(region_round((size_t)3 << 20, (size_t)2 << 20, &out) == 0 && out == (size_t)4 << 20);
  assert(region_round(0, 4096, &out) == -1);
  assert(region_round(SIZE_MAX, 4096, &out) == -1);
  assert(region_round(SIZE_MAX - 4095, 4096, &out) == 0 && out == SIZE_MAX - 4095);
  assert(region_round(SIZE_MAX - 4096, 4096, &out) == 0 && out % 4096 == 0);
  assert(region_round(1, 0, &out) == -1);
}

static void unit_portable_regions(void) {
  char err[256];
  NumaRegion r, none = {0};
  assert(region_reserve_try(&r, 0, 0, -1, ROLE_WORK, 0, err, sizeof(err)) == -1 && strstr(err, "invalid region size"));
  assert(region_reserve_try(&r, SIZE_MAX, 0, -1, ROLE_WORK, 0, err, sizeof(err)) == -1);
  assert(region_live_count() == 0);

  assert(region_reserve_try(&r, 100000, 0, -1, ROLE_KV_K, 0, err, sizeof(err)) == 0);
  assert(r.node == -1 && r.backend == BACKEND_PORTABLE && r.data_bytes >= 100000 && r.logical_bytes == 100000);
  assert(r.data >= r.mapping_base && (char *)r.data + r.data_bytes <= (char *)r.mapping_base + r.mapping_bytes);
  assert(region_live_count() == 1 && region_live_at(0)->data == r.data && !region_live_at(1));
  assert(memory_current[MEM_KV] == 0); /* reserved, not yet populated */
  memset(r.data, 7, r.logical_bytes);  /* anonymous memory is writable and zero-filled until then */
  region_mark_populated(&r);
  region_mark_populated(&r); /* counted once */
  assert(memory_current[MEM_KV] == 100000 && memory_peak[MEM_KV] >= 100000);
  region_release(&r);
  assert(memory_current[MEM_KV] == 0 && region_live_count() == 0 && !r.mapping_base);
  region_release(&r); /* second release is a no-op */
  region_release(&none);
  region_release(NULL);

  /* Zero fill and read-only protection. */
  assert(region_reserve_try(&r, 4096, 1, -1, ROLE_WEIGHTS, 0, err, sizeof(err)) == 0);
  assert(((unsigned char *)r.data)[0] == 0 && ((unsigned char *)r.data)[4095] == 0 && r.socket == 1);
  ((unsigned char *)r.data)[0] = 9;
  region_mark_populated(&r);
  assert(memory_current[MEM_WEIGHTS_OWNED] == 4096);
  region_protect(&r);
  assert(r.read_only && region_live_at(0)->read_only && ((unsigned char *)r.data)[0] == 9);
  region_release(&r);
  assert(memory_current[MEM_WEIGHTS_OWNED] == 0);

  /* A released-by-registry descriptor is not unmapped twice. */
  NumaRegion a, b, c;
  assert(!region_reserve_try(&a, 10, 0, -1, ROLE_PACKED, 0, err, sizeof(err)));
  assert(!region_reserve_try(&b, 10, 0, -1, ROLE_EMBEDDING, 0, err, sizeof(err)));
  assert(!region_reserve_try(&c, 10, 0, -1, ROLE_SCRATCH, 0, err, sizeof(err)));
  region_mark_populated(&b);
  assert(region_live_count() == 3 && memory_current[MEM_EMBEDDING] == 10);
  region_release(&b); /* the middle entry; the others stay valid */
  assert(region_live_count() == 2 && region_live_at(0)->data == a.data && region_live_at(1)->data == c.data);
  region_release_all();
  assert(region_live_count() == 0);
  region_release(&a);
  region_release(&c);
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);

#ifndef __linux__
  assert(region_reserve_try(&r, 4096, 0, 0, ROLE_WORK, 0, err, sizeof(err)) == -1 && strstr(err, "requires Linux"));
  assert(region_live_count() == 0);
#endif
}

static void unit_page_check(void) {
  int status[] = {0, 0, 1, -ENOENT, -EFAULT, 2, -EACCES};
  PageCheck check = {0};
  page_check_classify(status, 7, 0, &check);
  /* -ENOENT and -EFAULT (what an untouched anonymous page reports) are unpopulated pages; anything else is an error. */
  assert(check.queried == 7 && check.on_node == 2 && check.wrong_node == 2 && check.missing == 2 && check.errors == 1);
  page_check_classify(status, 0, 0, &check);
  assert(check.queried == 7);

  size_t plan[100], again[100];
  size_t count = page_sample_plan(10000, 512, 100, plan);
  assert(count > 40 && count <= 100 && plan[0] == 0 && plan[count - 1] == 9999);
  for (size_t i = 1; i < count; i++)
    assert(plan[i - 1] < plan[i] && plan[i] < 10000);
  for (size_t extent = 0; extent < 20; extent++) { /* both ends of every 2 MiB extent are always sampled */
    size_t first = extent * 512, last = first + 511 < 10000 ? first + 511 : 9999, seen = 0;
    for (size_t i = 0; i < count; i++)
      seen += plan[i] == first || plan[i] == last;
    assert(seen == (first == last ? 1 : 2));
  }
  assert(page_sample_plan(10000, 512, 100, again) == count && !memcmp(plan, again, count * sizeof(*plan)));
  count = page_sample_plan(10000, 512, 10, plan); /* fewer samples than endpoints: spread over the region */
  assert(count > 5 && count <= 10 && plan[0] == 0 && plan[count - 1] == 9999);
  assert(page_sample_plan(10000, 512, 1, plan) == 1 && plan[0] == 0);
  assert(page_sample_plan(1, 512, 8, plan) == 1 && plan[0] == 0);
  assert(page_sample_plan(0, 512, 8, plan) == 0 && page_sample_plan(8, 512, 0, plan) == 0);
}

static void unit_proc_parsers(void) {
  /* Two adjacent VMAs, each with one 2 MiB huge page; the first is 4 MiB long. */
  const char *smaps = "7f0000000000-7f0000400000 rw-p 00000000 00:00 0 \n"
                      "Size:               4096 kB\nAnonymous:          2048 kB\nAnonHugePages:      2048 kB\n"
                      "VmFlags: rd wr mr mw me ac hg\n"
                      "7f0000400000-7f0000600000 rw-p 00000000 00:00 0 \n"
                      "Size:               2048 kB\nAnonHugePages:      2048 kB\nVmFlags: rd wr\n"
                      "7f0000700000-7f0000800000 r--p 00000000 00:00 0 \nAnonHugePages:         0 kB\n";
  assert(smaps_thp_bytes(smaps, 0x7f0000000000, 0x7f0000400000) == (size_t)2 << 20); /* whole VMA */
  assert(smaps_thp_bytes(smaps, 0x7f0000200000, 0x7f0000500000) == (size_t)2 << 20); /* two half overlaps */
  assert(smaps_thp_bytes(smaps, 0x7f0000000000, 0x7f0000600000) == (size_t)4 << 20); /* merged extent */
  assert(smaps_thp_bytes(smaps, 0x7f0000600000, 0x7f0000700000) == 0);               /* gap */
  assert(smaps_thp_bytes("", 0, ~(uintptr_t)0) == 0 && smaps_thp_bytes("garbage\nAnonHugePages: 4 kB", 0, 1) == 0);

  const char *maps = "7f0000000000 default anon=4 dirty=4 N0=4 kernelpagesize_kB=4\n"
                     "7f0000400000 bind:2 anon=1024 dirty=1024 N2=1000 N3=24 kernelpagesize_kB=4\n"
                     "7f0000800000 default file=/lib/x mapped=1 N0=1\n";
  size_t pages[4];
  char policy[32];
  assert(numa_maps_lookup(maps, 0x7f0000500000, policy, sizeof(policy), pages, 4) == 0);
  assert(!strcmp(policy, "bind:2") && pages[2] == 1000 && pages[3] == 24 && pages[0] == 0);
  assert(numa_maps_lookup(maps, 0x7f0000000010, policy, sizeof(policy), pages, 4) == 0);
  assert(!strcmp(policy, "default") && pages[0] == 4);
  assert(numa_maps_lookup(maps, 0x7e0000000000, policy, sizeof(policy), pages, 4) == -1);
  assert(numa_maps_lookup("", 1, policy, sizeof(policy), pages, 4) == -1);
}

static void report_at_exit(void) {
  size_t current = 0;
  for (int i = 0; i < MEM_COUNT; i++)
    current += memory_current[i];
  printf("live=%d current=%zu\n", region_live_count(), current);
  fflush(stdout);
}

static void fail_during_startup(void) {
  atexit(report_at_exit); /* registered before the first region, so it runs after the registry's own cleanup */
  NumaRegion k, v, w;
  region_reserve(&k, 1 << 20, 0, -1, ROLE_KV_K, 0);
  region_mark_populated(&k);
  region_reserve(&v, 1 << 20, 0, -1, ROLE_KV_V, 0);
  region_reserve(&w, 1 << 16, 0, -1, ROLE_WORK, 0);
  fail("injected startup failure after %d regions", region_live_count());
}

static void bind_region(int node, size_t bytes, int thp) {
  NumaRegion r;
  char err[512];
  if (region_reserve_try(&r, bytes, 0, node, ROLE_WORK, thp, err, sizeof(err))) {
    fprintf(stderr, "[ERROR] %s\n", err);
    exit(1);
  }
  printf("bound %zu bytes to node %d thp_requested=%d\n", r.data_bytes, r.node, r.thp_requested);
  region_release(&r);
}

/* Total size and count of the process's mappings, to show that a failed startup leaves none behind. */
static void mapping_totals(size_t *count, size_t *bytes) {
  *count = *bytes = 0;
  FILE *f = fopen("/proc/self/maps", "r");
  char line[1024];
  while (f && fgets(line, sizeof(line), f)) {
    unsigned long long lo, hi;
    if (sscanf(line, "%llx-%llx", &lo, &hi) == 2) {
      (*count)++;
      *bytes += (size_t)(hi - lo);
    }
  }
  if (f)
    fclose(f);
}

/* maps NODE: reserve and bind a region, then fail a second reservation (injected EPERM), then release everything. */
static void leak_check(int node) {
  size_t count0, bytes0, count1, bytes1;
  mapping_totals(&count0, &bytes0);
  NumaRegion a, b;
  char err[512];
  if (region_reserve_try(&a, (size_t)64 << 20, 0, node, ROLE_WORK, 0, err, sizeof(err)))
    fail("first reservation failed: %s", err);
  setenv("LLAMA_NUMA_FAULT", "EPERM", 1);
  int failed = region_reserve_try(&b, (size_t)64 << 20, 0, node, ROLE_WORK, 0, err, sizeof(err));
  unsetenv("LLAMA_NUMA_FAULT");
  region_release_all();
  mapping_totals(&count1, &bytes1);
  printf("{\"second_failed\": %d, \"live_regions\": %d, \"maps_before\": %zu, \"maps_after\": %zu, "
         "\"mapped_bytes_before\": %zu, \"mapped_bytes_after\": %zu, \"error\": \"%s\"}\n",
         failed != 0, region_live_count(), count0, count1, bytes0, bytes1, err);
}

typedef struct {
  unsigned char *data;
  size_t bytes;
} FillJob;

static void fill_slice(const void *p, int worker, int workers) {
  const FillJob *job = p;
  size_t lo = job->bytes / (size_t)workers * (size_t)worker,
         hi = worker + 1 == workers ? job->bytes : lo + job->bytes / (size_t)workers;
  memset(job->data + lo, 1, hi - lo);
}

/* verify NODE BYTES none|half|all [thp] [samples=N] [expect=M] [threads=T]
 * Reserve on NODE (-1: portable), populate part of it from this thread, and report where the pages are. expect=M
 * checks against M instead of NODE (a negative control for a portable region under an external policy).
 * threads=T populates from a team of T workers pinned to the first T CPUs this process may run on. */
static void verify(int argc, char **argv) {
  int node = atoi(argv[2]), thp = 0, expect = node, threads = 1;
  size_t bytes = (size_t)strtoull(argv[3], NULL, 10), samples = 0;
  const char *touch = argv[4];
  for (int i = 5; i < argc; i++) {
    if (!strcmp(argv[i], "thp"))
      thp = 1;
    else if (!strncmp(argv[i], "samples=", 8))
      samples = (size_t)strtoull(argv[i] + 8, NULL, 10);
    else if (!strncmp(argv[i], "expect=", 7))
      expect = atoi(argv[i] + 7);
    else if (!strncmp(argv[i], "threads=", 8))
      threads = atoi(argv[i] + 8);
  }
  NumaRegion r;
  region_reserve(&r, bytes, 0, node, ROLE_WORK, thp);
  size_t populated = !strcmp(touch, "all") ? r.logical_bytes : !strcmp(touch, "half") ? r.logical_bytes / 2 : 0;
  int cpus[1024], allowed = numa_affinity_cpus(cpus, 1024);
  double start = monotonic_seconds();
  if (threads > 1) {
    if (allowed >= 0 && threads > allowed)
      threads = allowed;
    PoolConfig config = {.teams = 1, .threads = {threads, 0}, .cpus = {allowed > 0 ? cpus : NULL, NULL}};
    pool_start(&config);
    start = monotonic_seconds();
    FillJob job = {r.data, populated};
    team_run(pool_team(0), fill_slice, &job);
  } else
    memset(r.data, 1, populated);
  double populate_s = monotonic_seconds() - start;
  if (threads > 1)
    pool_stop();
  region_mark_populated(&r);
  NumaRegion check = r;
  check.node = expect;
  PageCheck pc;
  start = monotonic_seconds();
  int status = region_verify(&check, samples, &pc);
  double verify_s = monotonic_seconds() - start;
  size_t thp_bytes = 0, pages[64] = {0};
  char policy[64] = "";
  int have_thp = region_thp_bytes(&r, &thp_bytes) == 0,
      have_maps = region_numa_maps(&r, policy, sizeof(policy), pages, 64) == 0;
  printf("{\"node\": %d, \"expect\": %d, \"bytes\": %zu, \"populated_bytes\": %zu, \"populate_seconds\": %.6f, "
         "\"populate_threads\": %d, \"populate_gbps\": %.3f, \"verify_status\": %d, \"verify_seconds\": %.6f, "
         "\"full\": %d, \"scheme\": \"%s\", "
         "\"queried\": %zu, \"on_node\": %zu, \"wrong_node\": %zu, \"missing\": %zu, \"errors\": %zu, "
         "\"thp_requested\": %d, \"thp_bytes\": %s",
         node, expect, r.logical_bytes, populated, populate_s, threads,
         populate_s > 0 ? populated / populate_s / 1e9 : 0.0, status, verify_s, pc.full, pc.scheme, pc.queried,
         pc.on_node, pc.wrong_node, pc.missing, pc.errors, r.thp_requested, have_thp ? "" : "null");
  if (have_thp)
    printf("%zu", thp_bytes);
  printf(", \"numa_maps_policy\": ");
  if (have_maps) {
    printf("\"%s\", \"numa_maps_node_pages\": {", policy);
    for (int i = 0, first = 1; i < 64; i++)
      if (pages[i]) {
        printf("%s\"%d\": %zu", first ? "" : ", ", i, pages[i]);
        first = 0;
      }
    printf("}");
  } else
    printf("null");
  printf("}\n");
  region_release(&r);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    unit_rounding();
    unit_portable_regions();
    unit_page_check();
    unit_proc_parsers();
    return 0;
  }
  if (argc == 2 && !strcmp(argv[1], "fail3"))
    fail_during_startup();
  if (argc >= 5 && !strcmp(argv[1], "verify")) {
    verify(argc, argv);
    return 0;
  }
  if ((argc == 4 || argc == 5) && !strcmp(argv[1], "bind")) {
    bind_region(atoi(argv[2]), (size_t)strtoull(argv[3], NULL, 10), argc == 5 && !strcmp(argv[4], "thp"));
    return 0;
  }
  if (argc == 3 && !strcmp(argv[1], "maps")) {
    leak_check(atoi(argv[2]));
    return 0;
  }
  return 2;
}
