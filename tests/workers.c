/* Worker-team tests.
 *   test-workers                       lifecycle, coverage and barrier tests (barrier rounds from argv[1])
 *   test-workers barrier ROUNDS [N]    barrier stress with random delays on a team of N (default 5)
 *   test-workers fail-worker           a worker calls fail(): the process must exit, not hang
 *   test-workers fail-leader           the leader calls fail() with idle workers
 *   test-workers affinity [N]          pin N workers to allowed CPUs (Linux) and print the read-back as JSON
 *   test-workers bench-barrier [ROUNDS] per-barrier cost for team sizes 2..N (measurement, not a test) */
#include "workers.h"
#include "numa.h"
#include "runtime.h"
#include <assert.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  atomic_int visits[64];
  int workers_seen;
} VisitJob;

static void visit(const void *p, int worker, int workers) {
  VisitJob *job = (VisitJob *)p;
  assert(worker >= 0 && worker < workers && workers <= 64);
  atomic_fetch_add(&job->visits[worker], 1);
  if (worker == 0)
    job->workers_seen = workers;
}

static void unit_lifecycle(void) {
  assert(!pool_team(0));
  PoolConfig config = {.teams = 1, .threads = {4, 0}, .cpus = {NULL, NULL}};
  pool_start(&config);
  Team *team = pool_team(0);
  assert(team && team_size(team) == 4 && team_index(team) == 0 && !pool_team(1) && !pool_team(-1));
  assert(!team_affinity_verified(team));
  static VisitJob job;
  for (int run = 0; run < 2000; run++)
    team_run(team, visit, &job);
  for (int w = 0; w < 4; w++)
    assert(atomic_load(&job.visits[w]) == 2000); /* every member ran every job exactly once */
  assert(job.workers_seen == 4);
  pool_stop();
  pool_stop(); /* idempotent */
  assert(!pool_team(0));
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);

  /* Two teams are independent, and a team of one runs inline. */
  PoolConfig two = {.teams = 2, .threads = {3, 1}, .cpus = {NULL, NULL}};
  pool_start(&two);
  memset(&job, 0, sizeof(job));
  team_run(pool_team(1), visit, &job);
  assert(atomic_load(&job.visits[0]) == 1 && job.workers_seen == 1);
  team_run(pool_team(0), visit, &job);
  assert(job.workers_seen == 3 && team_index(pool_team(1)) == 1);
  team_barrier(pool_team(1)); /* single member: a no-op */
  pool_stop();
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
}

static unsigned char coverage[1000003];
static void mark(const void *job, int r0, int r1) {
  (void)job;
  assert(r0 >= 0 && r1 > r0);
  for (int r = r0; r < r1; r++)
    coverage[r]++;
}

static void unit_rows(void) {
  const int rows[] = {0, 1, 2, 5, 100, 1000003};
  for (int threads = 1; threads <= 7; threads += threads < 3 ? 1 : 2) {
    PoolConfig config = {.teams = 1, .threads = {threads, 0}, .cpus = {NULL, NULL}};
    pool_start(&config);
    for (size_t i = 0; i < sizeof(rows) / sizeof(*rows); i++) {
      memset(coverage, 0, sizeof(coverage));
      team_parallel_rows(pool_team(0), rows[i], mark, NULL);
      for (int r = 0; r < rows[i]; r++)
        assert(coverage[r] == 1); /* contiguous ranges that tile [0, rows) exactly */
      for (int r = rows[i]; r < rows[i] + 8 && r < 1000003; r++)
        assert(coverage[r] == 0);
    }
    pool_stop();
  }
}

typedef struct {
  int rounds;
  unsigned seed;
  _Atomic int slot[64];
  atomic_int errors;
  Team *team;
} BarrierJob;

static void delay(unsigned *state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  unsigned spins = *state % 97;
  if (spins < 3)
    sched_yield();
  else if (spins < 20)
    for (volatile unsigned i = 0; i < spins * 8; i++) {
    }
}

static void barrier_fn(const void *p, int worker, int workers) {
  BarrierJob *job = (BarrierJob *)p;
  unsigned state = job->seed + 2654435761u * (unsigned)(worker + 1);
  for (int round = 1; round <= job->rounds; round++) {
    delay(&state);
    atomic_store_explicit(&job->slot[worker], round, memory_order_relaxed);
    team_barrier(job->team);
    for (int w = 0; w < workers; w++)
      if (atomic_load_explicit(&job->slot[w], memory_order_relaxed) != round)
        atomic_fetch_add(&job->errors, 1);
    team_barrier(job->team); /* nobody writes round + 1 before everybody has read round */
  }
}

static void barrier_stress(int rounds, int threads) {
  PoolConfig config = {.teams = 1, .threads = {threads, 0}, .cpus = {NULL, NULL}};
  pool_start(&config);
  static BarrierJob job;
  job = (BarrierJob){.rounds = rounds, .seed = 12345, .team = pool_team(0)};
  team_run(pool_team(0), barrier_fn, &job);
  assert(atomic_load(&job.errors) == 0);
  pool_stop();
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
}

static void fail_fn(const void *job, int worker, int workers) {
  (void)job;
  (void)workers;
  if (worker == 1)
    fail("injected failure in worker %d", worker);
  for (;;)
    sched_yield(); /* the others never finish: only the failing thread can end the process */
}

static void affinity(int threads) {
  int cpus[1024];
  int allowed = numa_affinity_cpus(cpus, 1024);
  if (allowed < 1) {
    printf("{\"supported\": false}\n");
    return;
  }
  if (threads > allowed)
    threads = allowed;
  PoolConfig config = {.teams = 1, .threads = {threads, 0}, .cpus = {cpus, NULL}};
  pool_start(&config);
  Team *team = pool_team(0);
  printf("{\"supported\": true, \"verified\": %d, \"workers\": [", team_affinity_verified(team));
  for (int w = 0; w < threads; w++)
    printf("%s{\"assigned\": %d, \"observed\": %d}", w ? ", " : "", cpus[w], team_worker_cpu(team, w));
  printf("]}\n");
  pool_stop();
}

typedef struct {
  int rounds;
  Team *team;
} BenchJob;
static void bench_fn(const void *p, int worker, int workers) {
  (void)worker;
  (void)workers;
  const BenchJob *job = p;
  for (int i = 0; i < job->rounds; i++)
    team_barrier(job->team);
}

static void bench_barrier(int rounds, int max_threads) {
  for (int threads = 2;; threads *= 2) {
    if (threads > max_threads)
      threads = max_threads;
    PoolConfig config = {.teams = 1, .threads = {threads, 0}, .cpus = {NULL, NULL}};
    pool_start(&config);
    BenchJob job = {rounds, pool_team(0)};
    team_run(pool_team(0), bench_fn, &job); /* warm up */
    double start = monotonic_seconds();
    team_run(pool_team(0), bench_fn, &job);
    double seconds = monotonic_seconds() - start;
    printf("{\"threads\": %d, \"rounds\": %d, \"ns_per_barrier\": %.1f}\n", threads, rounds, seconds / rounds * 1e9);
    pool_stop();
    if (threads == max_threads)
      break;
  }
}

int main(int argc, char **argv) {
  if (argc == 1 || (argc == 2 && atoi(argv[1]) > 0)) {
    unit_lifecycle();
    unit_rows();
    barrier_stress(argc == 2 ? atoi(argv[1]) : 20000, 5);
    barrier_stress(20000, 2);
    barrier_stress(20000, 7);
    return 0;
  }
  if (argc >= 3 && !strcmp(argv[1], "barrier")) {
    barrier_stress(atoi(argv[2]), argc > 3 ? atoi(argv[3]) : 5);
    return 0;
  }
  if (argc == 2 && !strcmp(argv[1], "fail-worker")) {
    PoolConfig config = {.teams = 1, .threads = {3, 0}, .cpus = {NULL, NULL}};
    pool_start(&config);
    team_run(pool_team(0), fail_fn, NULL);
    return 3;
  }
  if (argc == 2 && !strcmp(argv[1], "fail-leader")) {
    PoolConfig config = {.teams = 1, .threads = {3, 0}, .cpus = {NULL, NULL}};
    pool_start(&config);
    fail("injected failure in the leader");
  }
  if (argc >= 2 && !strcmp(argv[1], "affinity")) {
    affinity(argc > 2 ? atoi(argv[2]) : 4);
    return 0;
  }
  if (argc >= 2 && !strcmp(argv[1], "bench-barrier")) {
    bench_barrier(argc > 2 ? atoi(argv[2]) : 200000, argc > 3 ? atoi(argv[3]) : 8);
    return 0;
  }
  return 2;
}
