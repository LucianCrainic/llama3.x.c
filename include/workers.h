#ifndef LLAMA_WORKERS_H
#define LLAMA_WORKERS_H

#include "numa.h"

/* Persistent worker teams: the only scheduler. Threads are created once (pool_start) and wait for jobs; the caller
 * of team_run is worker 0 of its team and does its share instead of sleeping. Workers synchronize with a
 * generation-counted barrier built on C11 atomics: they spin for a bounded time, then sched_yield. Workers must not
 * touch the engine's global accounting (memory_alloc, metrics); only the coordinator thread does. */
typedef struct Team Team;
typedef void (*RowFn)(const void *job, int r0, int r1);
typedef void (*TeamFn)(const void *job, int worker, int workers);

typedef struct {
  int teams;          /* 1 or 2 */
  int threads[2];     /* per team, leader included */
  const int *cpus[2]; /* one CPU id per thread, in worker order; NULL = unpinned */
} PoolConfig;

void pool_start(const PoolConfig *config); /* once, at load; fails on a bad configuration */
void pool_stop(void);                      /* joins every worker; a no-op when called from a pool thread */
Team *pool_team(int index);                /* NULL when not started */
int team_size(const Team *team);
int team_index(const Team *team); /* socket/team id */

/* Runs fn(job, worker, workers) on every member and returns after all have finished. Not reentrant. */
void team_run(Team *team, TeamFn fn, const void *job);
/* Splits [0, rows) into one contiguous, balanced range per member. */
void team_parallel_rows(Team *team, int rows, RowFn fn, const void *job);
/* Barrier among all members; only valid inside a TeamFn. */
void team_barrier(Team *team);

/* Pinning evidence. team_worker_cpu is the CPU the worker's affinity mask was read back as (-1 when it is not a
 * single CPU, or unpinned); team_affinity_verified is 1 when every worker was pinned and read back as assigned. */
int team_worker_cpu(const Team *team, int worker);
/* Pins the calling thread to the CPU of worker 0 of `team` (a no-op for an unpinned team) and returns the CPU it reads
 * back as, or -1. The leader of a team is whichever thread calls team_run, so each such thread must adopt its team:
 * pool_start pins its caller to every team in turn and leaves it on team 0, and a thread created later inherits that
 * single-CPU mask. The CPU the calling thread is running on (-1 when unknown) is cpu_current(). */
int team_adopt_leader(Team *team);
int cpu_current(void);

/* What the report records about each team. It is filled at pool_start and survives pool_stop. */
enum { TEAM_SUMMARY_THREADS = 1024 };
typedef struct {
  int index, threads, pinned, verified;
  int assigned[TEAM_SUMMARY_THREADS], observed[TEAM_SUMMARY_THREADS]; /* CPU ids; -1 unpinned or unknown */
} TeamSummary;
int pool_summary_count(void);
const TeamSummary *pool_summary(int index);
int team_affinity_verified(const Team *team);

/* ---- Cross-socket reduction (B4) -------------------------------------------------------------------------------
 * Combines the two teams' partial outputs and adds them to each team's own residual once:
 *   residual_s[i] += partial_0[i] + partial_1[i]      (the operand order is the same on both sockets)
 * Per reduction (generation g, counted per exchange from 1):
 *   1. the coordinator of socket s calls exchange_partial: it waits until the peer has consumed the buffer's previous
 *      occupant (generation g - 2), then returns partial_s[g & 1] for the local projection to write;
 *   2. the team meets a local barrier, the leader publishes ready_s = g (release);
 *   3. every worker waits for ready_peer >= g (acquire), then adds its disjoint element range of both partials;
 *   4. after a second local barrier the leader publishes consumed_s = g (release).
 * Each socket writes only its own residual. ready and consumed sit on separate 64-byte lines. The two sockets'
 * coordinators run concurrently (each is worker 0 of its own team, so team 1 needs a thread of its own to call
 * team_run). */
typedef struct Exchange Exchange;

/* Bytes a socket's partial buffers occupy; a region passed to exchange_create must hold at least this. */
size_t exchange_region_bytes(int elements_max);
/* Needs a started pool with two teams. socket_regions[s] (NULL, or NULL entries = heap) holds socket s's buffers
 * and is owned by the caller, who releases it only after exchange_destroy. */
Exchange *exchange_create(int elements_max, NumaRegion *const socket_regions[2]);
/* Waits for the last consumed acknowledgment of both sockets (or the abort flag), then frees the exchange. */
void exchange_destroy(Exchange *exchange);
/* Once per reduction, by the coordinator of `socket`, before the job that writes the partial; blocks until the
 * buffer is free. Returns NULL after an abort. Calls for one socket must alternate with exchange_reduce_add. */
float *exchange_partial(Exchange *exchange, int socket);
/* Called by every worker of `socket` inside a TeamFn, after the local projection wrote `elements` values to the
 * pointer exchange_partial returned. A no-op after an abort. */
void exchange_reduce_add(Exchange *exchange, int socket, int worker, int workers, float *residual, int elements);
/* Seconds (monotonic_seconds) at which one worker passed each step of exchange_reduce_add: left the first barrier, saw
 * the peer's partial, finished adding, left the closing barrier. For tracing (LLAMA_TP_TRACE); see tp.c. */
typedef struct {
  double after_barrier1, after_peer, after_add, after_barrier2;
} ReduceStamps;
/* exchange_reduce_add that also records the worker's step times in *stamps (NULL: none). */
void exchange_reduce_add_stamped(Exchange *exchange, int socket, int worker, int workers, float *residual, int elements,
                                 ReduceStamps *stamps);
/* Flag form of the reduction (the fused projection, LLAMA_TP_REDUCE=flags): no team barrier at all. Each socket keeps one
 * generation flag per output row next to its partial buffers. A member that finished output rows [r0, r1) of the partial
 * calls exchange_rows_done (the kernels' RowsDoneFn, with the Exchange and socket as context via a thin wrapper); then,
 * for its own add range, exchange_reduce_add_flags waits for the flags of those rows of BOTH sockets and adds, so it
 * starts as soon as the producers of its rows are done and never waits for the rest of the team. Partial buffer reuse is
 * still guarded by `consumed`, which the socket's coordinator publishes with exchange_reduce_finish after the team job
 * has joined (every add of the generation is then finished). Same arithmetic and operand order as exchange_reduce_add.
 * The generation is that of the last exchange_partial of the socket. Do not mix the two forms within one generation. */
void exchange_rows_done(Exchange *exchange, int socket, int r0, int r1);
void exchange_reduce_add_flags(Exchange *exchange, int socket, int worker, int workers, float *residual, int elements,
                               ReduceStamps *stamps);
void exchange_reduce_finish(Exchange *exchange, int socket);
/* Process-wide and irreversible in production (it is workers_abort): every wait loop of the exchange and of the
 * team barriers returns, and the results of the generation in flight are garbage. */
void exchange_abort(Exchange *exchange);
/* Seconds the leader of `socket` spent adding (step 3) and waiting (steps 2-4 barriers and ready), when timing is on.
 */
void exchange_set_timing(Exchange *exchange, int on);
void exchange_timing(const Exchange *exchange, int socket, double *reduce_seconds, double *wait_seconds);
/* The same split further: seconds[0] waiting for the team's last partial write (the slowest member of the socket),
 * [1] waiting for the peer socket's partial, [2] adding, [3] the closing barrier, [4] waiting for a free partial buffer
 * in exchange_partial. Leader only. Read it only while no reduction is running. */
void exchange_timing_detail(const Exchange *exchange, int socket, double seconds[5]);

/* fail() sets this before running exit handlers, so no wait loop outlives the memory it spins on. After it, stop
 * the pool; barrier and exchange state is not consistent any more. workers_abort_clear is for tests only. */
void workers_abort(void);
int workers_aborted(void);
void workers_abort_clear(void);

#endif
