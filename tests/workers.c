/* Worker-team tests.
 *   test-workers                       lifecycle, coverage and barrier tests (barrier rounds from argv[1])
 *   test-workers barrier ROUNDS [N]    barrier stress with random delays on a team of N (default 5)
 *   test-workers fail-worker           a worker calls fail(): the process must exit, not hang
 *   test-workers fail-leader           the leader calls fail() with idle workers
 *   test-workers affinity [N]          pin N workers to allowed CPUs (Linux) and print the read-back as JSON
 *   test-workers bench-barrier [ROUNDS] per-barrier cost for team sizes 2..N (measurement, not a test)
 *   test-workers exchange GENS [T0 T1 E]   cross-socket reduction stress, bitwise against a serial reference
 *   test-workers bench-exchange [GENS] [T] one reduction's latency (median, p99) for decode and prefill sizes
 *   test-workers exchange-fail-worker      a socket-1 worker calls fail() mid-generation: exit, not hang
 *   test-workers exchange-fail-leader      the socket-0 coordinator calls fail() while socket 1 waits for it */
#include "workers.h"
#include "numa.h"
#include "runtime.h"
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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

/* ---- Cross-socket reduction (B4) --------------------------------------------------------------------------------
 * Two coordinator threads (the main thread for socket 0, a pthread for socket 1) drive two teams through G
 * generations. Every generation's partials are a function of (seed, g, socket, i), so a buffer overwritten before the
 * peer consumed it, or read before it was ready, changes some residual bit. */
static unsigned hash32(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}
static float partial_value(unsigned seed, unsigned g, int socket, int i) {
  unsigned h = hash32(seed ^ hash32(g * 0x9E3779B1U + (unsigned)socket * 0x85EBCA6BU + (unsigned)i * 0xC2B2AE35U));
  return (float)(int)h * (1.0f / 2147483648.0f);
}

typedef struct {
  Exchange *exchange;
  int socket, elements, generations;
  unsigned seed;
  unsigned delay_permille, delay_us; /* chance per protocol step and longest sleep */
  int fail_at, fail_worker;          /* generation at which to call fail(); fail_worker -1 = the coordinator */
  int flags;                         /* the flag form: rows produced in interleaved chunks, no team barrier */
  float *residual;
} Side;

typedef struct {
  Side *side;
  unsigned generation;
  float *partial;
} GenJob;

static void inject(unsigned *state, const Side *side) {
  *state = hash32(*state + 0x6D2B79F5U);
  if (*state % 1000 < side->delay_permille)
    usleep(*state / 1000 % (side->delay_us + 1));
  else if (*state % 11 == 0)
    sched_yield();
}

static void generation_fn(const void *p, int worker, int workers) {
  const GenJob *job = p;
  Side *side = job->side;
  unsigned state = side->seed + 0x9E3779B9U * (unsigned)(worker + 1) + job->generation * 7919U;
  int i0 = (int)((long long)side->elements * worker / workers),
      i1 = (int)((long long)side->elements * (worker + 1) / workers);
  inject(&state, side);
  if ((int)job->generation == side->fail_at && worker == side->fail_worker)
    fail("injected failure in exchange worker %d of socket %d", worker, side->socket);
  if (side->flags) {
    /* Rows in chunks of 5, dealt to the members differently on each socket and generation, as claiming would. */
    const int chunk = 5, chunks = (side->elements + chunk - 1) / chunk;
    for (int c = 0; c < chunks; c++) {
      if ((c + (int)job->generation + side->socket) % workers != worker)
        continue;
      int r0 = c * chunk, r1 = r0 + chunk < side->elements ? r0 + chunk : side->elements;
      for (int i = r0; i < r1; i++)
        job->partial[i] = partial_value(side->seed, job->generation, side->socket, i);
      inject(&state, side);
      exchange_rows_done(side->exchange, side->socket, r0, r1);
    }
    exchange_reduce_add_flags(side->exchange, side->socket, worker, workers, side->residual, side->elements, NULL);
    inject(&state, side);
    return;
  }
  for (int i = i0; i < i1; i++)
    job->partial[i] = partial_value(side->seed, job->generation, side->socket, i);
  inject(&state, side); /* a late ready */
  exchange_reduce_add(side->exchange, side->socket, worker, workers, side->residual, side->elements);
  inject(&state, side); /* a late next projection */
}

static void run_side(Side *side) {
  Team *team = pool_team(side->socket);
  unsigned state = side->seed ^ 0xA5A5A5A5U;
  for (int g = 1; g <= side->generations; g++) {
    inject(&state, side);
    if (g == side->fail_at && side->fail_worker < 0)
      fail("injected failure in the coordinator of socket %d", side->socket);
    float *partial = exchange_partial(side->exchange, side->socket);
    if (!partial)
      return; /* aborted */
    GenJob job = {side, (unsigned)g, partial};
    team_run(team, generation_fn, &job);
    if (side->flags)
      exchange_reduce_finish(side->exchange, side->socket);
  }
}

static void *side_main(void *p) {
  run_side(p);
  return NULL;
}

typedef struct {
  int threads[2], elements, generations;
  unsigned delay_permille[2], delay_us[2];
  int use_regions, fail_at, fail_worker, flags;
} ExchangeCase;

static void exchange_case(const ExchangeCase *c) {
  PoolConfig config = {.teams = 2, .threads = {c->threads[0], c->threads[1]}, .cpus = {NULL, NULL}};
  pool_start(&config);
  NumaRegion regions[2] = {{0}, {0}};
  NumaRegion *use[2] = {NULL, NULL};
  if (c->use_regions)
    for (int s = 0; s < 2; s++) {
      region_reserve(&regions[s], exchange_region_bytes(c->elements), s, -1, ROLE_WORK, 0);
      use[s] = &regions[s];
    }
  Exchange *e = exchange_create(c->elements, use);
  float *initial = memory_alloc((size_t)c->elements, sizeof(float), MEM_ACTIVATION);
  float *reference = memory_alloc((size_t)c->elements, sizeof(float), MEM_ACTIVATION);
  Side side[2];
  for (int i = 0; i < c->elements; i++)
    initial[i] = partial_value(99, 0, 2, i);
  memcpy(reference, initial, (size_t)c->elements * sizeof(float));
  for (int s = 0; s < 2; s++) {
    side[s] = (Side){.exchange = e,
                     .socket = s,
                     .elements = c->elements,
                     .generations = c->generations,
                     .seed = 1234,
                     .delay_permille = c->delay_permille[s],
                     .delay_us = c->delay_us[s],
                     .fail_at = c->fail_at,
                     .fail_worker = c->fail_worker,
                     .flags = c->flags,
                     .residual = memory_alloc((size_t)c->elements, sizeof(float), MEM_ACTIVATION)};
    side[s].fail_at = c->fail_at && ((s == 1) == (c->fail_worker >= 0)) ? c->fail_at : 0; /* one socket fails */
    memcpy(side[s].residual, initial, (size_t)c->elements * sizeof(float));
  }
  pthread_t thread;
  assert(!pthread_create(&thread, NULL, side_main, &side[1]));
  if (c->fail_at)
    pthread_detach(thread); /* fail() ends the process; nothing joins the coordinator of socket 1 */
  run_side(&side[0]);
  if (!c->fail_at)
    pthread_join(thread, NULL);
  for (int g = 1; g <= c->generations; g++)
    for (int i = 0; i < c->elements; i++)
      reference[i] += partial_value(1234, (unsigned)g, 0, i) + partial_value(1234, (unsigned)g, 1, i);
  for (int s = 0; s < 2; s++) {
    for (int i = 0; i < c->elements; i++)
      if (memcmp(&side[s].residual[i], &reference[i], sizeof(float))) {
        fprintf(stderr, "socket %d element %d differs from the serial reference\n", s, i);
        abort();
      }
    memory_free(side[s].residual);
  }
  memory_free(reference);
  memory_free(initial);
  exchange_destroy(e);
  for (int s = 0; s < 2; s++)
    region_release(&regions[s]);
  pool_stop();
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
}

static void unit_exchange(int scale) {
  const int teams[] = {1, 3, 7};
  const int sizes[] = {1, 7, 4096, 4096 * 64};
  for (int flags = 0; flags < 2; flags++)
    for (size_t t = 0; t < sizeof(teams) / sizeof(*teams); t++)
      for (size_t k = 0; k < sizeof(sizes) / sizeof(*sizes); k++) {
        ExchangeCase c = {.threads = {teams[t], teams[t]},
                          .elements = sizes[k],
                          .generations = 1000 * scale / 100,
                          .fail_worker = -1,
                          .flags = flags};
        if (sizes[k] > 4096)
          c.generations /= 5;
        c.use_regions = (int)((t + k) & 1);
        exchange_case(&c);
      }
  for (int flags = 0; flags < 2; flags++) {
    ExchangeCase skewed = {.threads = {3, 1}, .elements = 4096, .generations = 300 * scale / 100, .fail_worker = -1,
                           .flags = flags};
    exchange_case(&skewed);
    for (int slow = 0; slow < 2; slow++) {
      ExchangeCase c = {.threads = {3, 3}, .elements = 257, .generations = 400 * scale / 100, .fail_worker = -1,
                        .flags = flags};
      c.delay_permille[slow] = 300;
      c.delay_us[slow] = 300;
      exchange_case(&c);
    }
    ExchangeCase mixed = {.threads = {2, 5}, .elements = 1000, .generations = 400 * scale / 100, .fail_worker = -1,
                          .flags = flags, .delay_permille = {150, 150}, .delay_us = {100, 400}};
    exchange_case(&mixed);
  }
}

typedef struct {
  Exchange *exchange;
  atomic_int done;
} AbortSide;

static void *abort_later(void *p) {
  usleep(50000);
  exchange_abort(((AbortSide *)p)->exchange);
  return NULL;
}

static void unit_exchange_abort(void) {
  PoolConfig config = {.teams = 2, .threads = {3, 3}, .cpus = {NULL, NULL}};
  pool_start(&config);
  Exchange *e = exchange_create(64, NULL);
  float residual[64] = {0};
  Side side = {
      .exchange = e, .socket = 0, .elements = 64, .generations = 1, .seed = 5, .fail_worker = -1, .residual = residual};
  AbortSide abort_side = {.exchange = e};
  pthread_t thread;
  /* Socket 1 never publishes: socket 0's workers wait for ready_1 until the abort releases them. */
  assert(!pthread_create(&thread, NULL, abort_later, &abort_side));
  run_side(&side);
  pthread_join(thread, NULL);
  assert(workers_aborted());
  /* A coordinator blocked on the reuse rule is released too, and gets no buffer. */
  workers_abort_clear();
  assert(exchange_partial(e, 1) && exchange_partial(e, 1)); /* generations 1 and 2 do not wait */
  assert(!pthread_create(&thread, NULL, abort_later, &abort_side));
  assert(exchange_partial(e, 1) == NULL); /* generation 3 waits for consumed_0 >= 1 */
  pthread_join(thread, NULL);
  exchange_destroy(e); /* returns despite the unacknowledged generations */
  /* The flag form: socket 0's members wait for the peer's row flags, which never come, until the abort releases them. */
  workers_abort_clear();
  Exchange *e2 = exchange_create(64, NULL);
  Side flagged = {.exchange = e2, .socket = 0, .elements = 64, .generations = 1, .seed = 5, .fail_worker = -1,
                  .flags = 1, .residual = residual};
  AbortSide abort_flagged = {.exchange = e2};
  assert(!pthread_create(&thread, NULL, abort_later, &abort_flagged));
  run_side(&flagged);
  pthread_join(thread, NULL);
  assert(workers_aborted());
  exchange_destroy(e2);
  pool_stop();
  workers_abort_clear();
  for (int i = 0; i < MEM_COUNT; i++)
    assert(memory_current[i] == 0);
}

static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

/* Latency of one reduction as seen by socket 0's coordinator: partial write, both barriers, the peer wait and the add.
 * Workers are pinned to the sockets' CPUs when the topology has two real sockets with enough of them. */
static void bench_exchange(int generations, int threads) {
  const Topology *topology = numa_topology();
  const int *cpus[2] = {NULL, NULL};
  if (topology->nsockets == 2 && topology->socket[0].cpus && topology->socket[1].cpus &&
      topology->socket[0].ncpus >= threads && topology->socket[1].ncpus >= threads) {
    cpus[0] = topology->socket[0].cpus;
    cpus[1] = topology->socket[1].cpus;
  }
  const int sizes[] = {4096, 4096 * 16, 4096 * 64, 4096 * 256};
  for (size_t k = 0; k < sizeof(sizes) / sizeof(*sizes); k++) {
    PoolConfig config = {.teams = 2, .threads = {threads, threads}, .cpus = {cpus[0], cpus[1]}};
    pool_start(&config);
    NumaRegion regions[2] = {{0}, {0}};
    NumaRegion *use[2] = {&regions[0], &regions[1]};
    for (int s = 0; s < 2; s++) {
      int node = topology->nsockets == 2 && cpus[0] ? topology->socket[s].hbm_node : -1;
      region_reserve(&regions[s], exchange_region_bytes(sizes[k]), s, node, ROLE_WORK, 0);
    }
    Exchange *e = exchange_create(sizes[k], use);
    Side side[2];
    for (int s = 0; s < 2; s++)
      side[s] = (Side){.exchange = e,
                       .socket = s,
                       .elements = sizes[k],
                       .generations = generations,
                       .seed = 7,
                       .fail_worker = -1,
                       .residual = memory_alloc((size_t)sizes[k], sizeof(float), MEM_ACTIVATION)};
    double *samples = memory_alloc((size_t)generations, sizeof(double), MEM_ACTIVATION);
    pthread_t thread;
    assert(!pthread_create(&thread, NULL, side_main, &side[1]));
    Team *team = pool_team(0);
    for (int g = 1; g <= generations; g++) {
      double start = monotonic_seconds();
      float *partial = exchange_partial(e, 0);
      GenJob job = {&side[0], (unsigned)g, partial};
      team_run(team, generation_fn, &job);
      samples[g - 1] = monotonic_seconds() - start;
    }
    pthread_join(thread, NULL);
    qsort(samples, (size_t)generations, sizeof(double), compare_double);
    printf("{\"threads\": %d, \"elements\": %d, \"pinned\": %d, \"generations\": %d, \"median_us\": %.2f, "
           "\"p99_us\": %.2f}\n",
           threads, sizes[k], cpus[0] != NULL, generations, samples[generations / 2] * 1e6,
           samples[(size_t)generations * 99 / 100] * 1e6);
    memory_free(samples);
    for (int s = 0; s < 2; s++)
      memory_free(side[s].residual);
    exchange_destroy(e);
    for (int s = 0; s < 2; s++)
      region_release(&regions[s]);
    pool_stop();
  }
}

#ifdef __linux__
/* Two pinned one-thread teams on two allowed CPUs: pool_start must leave its caller on team 0's CPU, and a thread that
 * adopts team 1 must run on team 1's CPU (it would otherwise share the single CPU it inherited from its creator). */
static void *adopt_main(void *p) {
  int *cpu = p;
  *cpu = team_adopt_leader(pool_team(1));
  sched_yield();
  cpu[1] = cpu_current();
  return NULL;
}

static void unit_leaders(void) {
  int allowed[1024];
  int n = numa_affinity_cpus(allowed, 1024);
  if (n < 2) {
    printf("leaders: skipped, fewer than two allowed CPUs\n");
    return;
  }
  int c0[1] = {allowed[0]}, c1[1] = {allowed[n - 1]};
  PoolConfig config = {.teams = 2, .threads = {1, 1}, .cpus = {c0, c1}};
  pool_start(&config);
  assert(cpu_current() == c0[0]);
  assert(team_worker_cpu(pool_team(0), 0) == c0[0]);
  int seen[2] = {-2, -2};
  pthread_t thread;
  assert(!pthread_create(&thread, NULL, adopt_main, seen));
  pthread_join(thread, NULL);
  assert(seen[0] == c1[0] && seen[1] == c1[0]);
  assert(cpu_current() == c0[0]);
  pool_stop();
}
#endif

int main(int argc, char **argv) {
  if (argc >= 2 && !strcmp(argv[1], "bench-exchange")) {
    bench_exchange(argc > 2 ? atoi(argv[2]) : 20000, argc > 3 ? atoi(argv[3]) : 4);
    return 0;
  }
  if (argc >= 2 && !strcmp(argv[1], "exchange")) {
    ExchangeCase c = {.threads = {argc > 3 ? atoi(argv[3]) : 3, argc > 4 ? atoi(argv[4]) : 3},
                      .elements = argc > 5 ? atoi(argv[5]) : 4096,
                      .generations = argc > 2 ? atoi(argv[2]) : 1000,
                      .fail_worker = -1,
                      .delay_permille = {50, 50},
                      .delay_us = {200, 200}};
    exchange_case(&c);
    return 0;
  }
  if (argc == 2 && (!strcmp(argv[1], "exchange-fail-worker") || !strcmp(argv[1], "exchange-fail-leader"))) {
    ExchangeCase c = {.threads = {3, 3}, .elements = 4096, .generations = 100, .fail_at = 17};
    c.fail_worker = !strcmp(argv[1], "exchange-fail-worker") ? 1 : -1;
    c.delay_permille[0] = c.delay_permille[1] = 50;
    c.delay_us[0] = c.delay_us[1] = 100;
    exchange_case(&c);
    return 3; /* unreachable: fail() exits */
  }
  if (argc == 1 || (argc == 2 && atoi(argv[1]) > 0)) {
    unit_exchange(100);
    unit_exchange_abort();
    unit_lifecycle();
    unit_rows();
#ifdef __linux__
    unit_leaders();
#endif
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
