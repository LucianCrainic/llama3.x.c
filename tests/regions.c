/* Bound-region tests and probe.
 *   test-regions                    unit tests (portable backend everywhere; bound backend notes on Linux)
 *   test-regions fail3              reserve three regions, then fail(); prints what the exit path left behind
 *   test-regions bind NODE BYTES    reserve a region bound to NODE; exit 1 with the message on failure
 *   test-regions probe ...          see probe() */
#include "numa.h"
#include "runtime.h"
#include <assert.h>
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

static void bind_region(int node, size_t bytes) {
  NumaRegion r;
  char err[512];
  if (region_reserve_try(&r, bytes, 0, node, ROLE_WORK, 0, err, sizeof(err))) {
    fprintf(stderr, "[ERROR] %s\n", err);
    exit(1);
  }
  printf("bound %zu bytes to node %d\n", r.data_bytes, r.node);
  region_release(&r);
}

int main(int argc, char **argv) {
  if (argc == 1) {
    unit_rounding();
    unit_portable_regions();
    return 0;
  }
  if (argc == 2 && !strcmp(argv[1], "fail3"))
    fail_during_startup();
  if (argc == 4 && !strcmp(argv[1], "bind")) {
    bind_region(atoi(argv[2]), (size_t)strtoull(argv[3], NULL, 10));
    return 0;
  }
  return 2;
}
