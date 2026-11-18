/* Attention microbenchmark (E0): the engine's batched prefill attention and its single-row decode attention on one
 * layer, with K/V in a region bound to one NUMA node. One JSON object per line on stdout.
 *   attention-bench --mode prefill [--shape 8b|1b|H/KV/HD,...] [--prompt 512,2048,8192] [--m 16,64,256] [--reps R]
 *   attention-bench --mode decode  [--shape ...] [--context 8192,32768] [--reps R]
 * Common: [--node N] [--kv-heads-local N] [--seq-extra N] [--portable] [--cold-mib N]
 * Prefill runs the whole prompt of one layer as src/run.c does (microbatches at pos0 = 0, M, 2M, ...; the workspace of
 * k_attention_init_rows with rows = M; k_attention_batch), so ms_layer * layers is comparable with the engine's profiled
 * attention time. Decode calls k_attention at the last position of a context, cycling through enough copies of K/V
 * (--cold-mib, default 1024) that each call reads memory the previous one did not leave in the cache.
 * Threads, pinning and the instruction set come from the engine's own environment: LLAMA_BACKEND=teams LLAMA_THREADS=56
 * LLAMA_TOPOLOGY (under numactl --cpunodebind=0), LLAMA_KERNELS, LLAMA_ATTN_ROW_BLOCK, LLAMA_ATTN_WORKSPACE_MIB.
 * One extra pass per cell runs with the kernels' task hook on (k_attention_set_task_hook) to measure how the workers are
 * used: the timed repetitions run without it. --portable runs the batched driver with the scalar chunk kernel
 * (k_attention_batch_portable), the only way to see tasks on a host without AVX-512. */
#include "kernels.h"
#include "numa.h"
#include "workers.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_TRACKS = 256, MAX_LIST = 32, MAX_REPS = 1024 };

static unsigned long long rng = 88172645463325252ull;
static float uniform(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (float)((double)(rng >> 40) / (double)(1ull << 24)) * 2.0f - 1.0f;
}

static int parse_list(const char *text, int *out, int max) {
  int n = 0;
  char *copy = strdup(text);
  for (char *token = strtok(copy, ","); token && n < max; token = strtok(NULL, ","))
    out[n++] = atoi(token);
  free(copy);
  return n;
}

static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

/* ---- task trace: one record per task and worker, no sharing between workers ------------------------------------------ */

typedef struct {
  double begin, end;
  int call, wave;
} Record;

typedef struct {
  Record *record;
  size_t count, capacity;
  double span_positions, scored_positions;
  char pad[24];
} Track;

typedef struct {
  _Alignas(64) Track track[MAX_TRACKS];
  atomic_int call;
  atomic_int dropped;
} Trace;

static void on_task(void *ctx, const AttentionTaskInfo *info) {
  Trace *trace = ctx;
  if (info->worker < 0 || info->worker >= MAX_TRACKS) {
    atomic_fetch_add_explicit(&trace->dropped, 1, memory_order_relaxed);
    return;
  }
  Track *track = &trace->track[info->worker];
  if (track->count == track->capacity) {
    track->capacity = track->capacity ? 2 * track->capacity : 1 << 14;
    track->record = realloc(track->record, track->capacity * sizeof(Record));
    if (!track->record)
      fail("out of memory for the task trace");
  }
  track->record[track->count++] =
      (Record){info->begin, info->end, atomic_load_explicit(&trace->call, memory_order_relaxed), info->wave};
  track->span_positions += info->span;
  track->scored_positions += (double)info->positions;
}

static void trace_reset(Trace *trace) {
  for (int i = 0; i < MAX_TRACKS; i++) {
    trace->track[i].count = 0;
    trace->track[i].span_positions = trace->track[i].scored_positions = 0;
  }
  atomic_store(&trace->call, 0);
  atomic_store(&trace->dropped, 0);
}

typedef struct {
  long tasks, waves;
  double span_positions;        /* K/V positions streamed, summed over tasks */
  double busy_mean, busy_min;   /* share of the task phases a worker spent inside tasks */
  double tail;                  /* share of the task phases a worker idled after its last task (H4) */
  double task_phase;            /* the task phases' share of the traced pass (the rest: merge, barriers, dispatch) */
  int valid;
} TraceStats;

typedef struct {
  double begin, end;
  int call, wave, worker;
} Flat;

static int compare_flat(const void *a, const void *b) {
  const Flat *x = a, *y = b;
  if (x->call != y->call)
    return x->call < y->call ? -1 : 1;
  if (x->wave != y->wave)
    return x->wave < y->wave ? -1 : 1;
  return (x->begin > y->begin) - (x->begin < y->begin);
}

/* Groups the records into waves (one team job each) and measures how the workers shared each task phase. */
static TraceStats trace_stats(const Trace *trace, int workers, double pass_seconds) {
  TraceStats s = {0};
  size_t total = 0;
  for (int i = 0; i < MAX_TRACKS; i++)
    total += trace->track[i].count;
  if (!total || workers < 1)
    return s;
  Flat *flat = malloc(total * sizeof(Flat));
  size_t n = 0;
  for (int w = 0; w < MAX_TRACKS; w++) {
    const Track *track = &trace->track[w];
    s.span_positions += track->span_positions;
    for (size_t i = 0; i < track->count; i++)
      flat[n++] = (Flat){track->record[i].begin, track->record[i].end, track->record[i].call, track->record[i].wave, w};
  }
  qsort(flat, n, sizeof(Flat), compare_flat);
  double *busy_total = calloc(MAX_TRACKS, sizeof(double)), *busy = calloc(MAX_TRACKS, sizeof(double)),
         *last = calloc(MAX_TRACKS, sizeof(double));
  double phases = 0, idle_tail = 0;
  for (size_t i = 0; i < n;) {
    size_t j = i;
    double first = flat[i].begin, finish = flat[i].end;
    for (; j < n && flat[j].call == flat[i].call && flat[j].wave == flat[i].wave; j++) {
      finish = flat[j].end > finish ? flat[j].end : finish;
      first = flat[j].begin < first ? flat[j].begin : first;
    }
    memset(busy, 0, MAX_TRACKS * sizeof(double));
    for (int w = 0; w < MAX_TRACKS; w++)
      last[w] = first;
    for (size_t k = i; k < j; k++) {
      busy[flat[k].worker] += flat[k].end - flat[k].begin;
      last[flat[k].worker] = flat[k].end > last[flat[k].worker] ? flat[k].end : last[flat[k].worker];
    }
    for (int w = 0; w < workers; w++) {
      busy_total[w] += busy[w];
      idle_tail += finish - last[w];
    }
    phases += finish - first;
    s.waves++;
    i = j;
  }
  s.tasks = (long)n;
  double sum = 0, least = 1e30;
  for (int w = 0; w < workers; w++) {
    sum += busy_total[w];
    least = busy_total[w] < least ? busy_total[w] : least;
  }
  if (phases > 0) {
    s.busy_mean = sum / (workers * phases);
    s.busy_min = least / phases;
    s.tail = idle_tail / (workers * phases);
    s.task_phase = pass_seconds > 0 ? phases / pass_seconds : 0;
    s.valid = 1;
  }
  free(flat), free(busy_total), free(busy), free(last);
  return s;
}

/* ---- cell setup -------------------------------------------------------------------------------------------------------- */

typedef struct {
  const char *name;
  int heads, kv_heads, head_dim, kv_model;
} Shape;

static int parse_shape(const char *text, Shape *shape) {
  int h, kv, hd;
  if (!strcmp(text, "8b"))
    h = 32, kv = 8, hd = 128;
  else if (!strcmp(text, "1b"))
    h = 32, kv = 8, hd = 64;
  else if (sscanf(text, "%d/%d/%d", &h, &kv, &hd) != 3 || h < 1 || kv < 1 || hd < 1 || h % kv)
    return -1;
  shape->name = strdup(text), shape->heads = h, shape->kv_heads = kv, shape->head_dim = hd, shape->kv_model = kv;
  return 0;
}

static void *carve(NumaRegion *region, size_t bytes, RegionRole role, int node) {
  region_reserve(region, bytes, 0, node, role, 0);
  return region->data;
}

static void json_list(const char *name, const double *v, int n, double scale) {
  printf("\"%s\": [", name);
  for (int i = 0; i < n; i++)
    printf("%s%.6f", i ? ", " : "", v[i] * scale);
  printf("]");
}

static void json_stats(const TraceStats *t, double kv_bytes_per_position) {
  if (!t->valid) {
    printf("\"tasks\": 0, \"waves\": 0, \"streamed_kv_bytes\": null, \"busy_mean\": null, \"busy_min\": null, "
           "\"tail_fraction\": null, \"task_phase_fraction\": null");
    return;
  }
  printf("\"tasks\": %ld, \"waves\": %ld, \"streamed_kv_bytes\": %.0f, \"busy_mean\": %.4f, \"busy_min\": %.4f, "
         "\"tail_fraction\": %.4f, \"task_phase_fraction\": %.4f",
         t->tasks, t->waves, t->span_positions * kv_bytes_per_position, t->busy_mean, t->busy_min, t->tail,
         t->task_phase);
}

typedef struct {
  Shape shape;
  int heads, kv_heads, hd, node, portable;
} Setup;

static void common_fields(const Setup *s, const char *mode, int workers) {
  printf("{\"mode\": \"%s\", \"isa\": \"%s\", \"backend\": \"%s\", \"workers\": %d, \"node\": %d, \"shape\": \"%s\", "
         "\"heads\": %d, \"kv_heads\": %d, \"head_dim\": %d, \"kv_heads_model\": %d, \"row_block\": %d, \"portable\": %s, ",
         mode, kernels_name(), kernels_backend_name(), workers, s->node, s->shape.name, s->heads, s->kv_heads, s->hd,
         s->shape.kv_model, k_attention_row_block(), s->portable ? "true" : "false");
}

static void prefill_cell(const Setup *s, int prompt, int m, int reps, int seq_extra) {
  int seq = prompt + seq_extra, heads = s->heads, kv_heads = s->kv_heads, hd = s->hd;
  size_t dim = (size_t)heads * hd, kv_dim = (size_t)kv_heads * hd;
  NumaRegion kr, vr, qr, wr;
  float *kc = carve(&kr, (size_t)seq * kv_dim * sizeof(float), ROLE_KV_K, s->node);
  float *vc = carve(&vr, (size_t)seq * kv_dim * sizeof(float), ROLE_KV_V, s->node);
  for (size_t i = 0; i < (size_t)seq * kv_dim; i++)
    kc[i] = uniform(), vc[i] = uniform();
  region_mark_populated(&kr), region_mark_populated(&vr);
  float *q = carve(&qr, ((size_t)2 * m * dim + (size_t)heads * seq) * sizeof(float), ROLE_SCRATCH, s->node);
  float *out = q + (size_t)m * dim, *att = out + (size_t)m * dim;
  for (size_t i = 0; i < (size_t)m * dim; i++)
    q[i] = uniform();
  region_mark_populated(&qr);
  size_t work_bytes = k_attention_bytes_rows(heads, kv_heads, hd, seq, m);
  AttentionWork work;
  float *memory = carve(&wr, work_bytes, ROLE_WORK, s->node);
  k_attention_init_rows(&work, heads, kv_heads, hd, seq, m, memory);
  region_mark_populated(&wr);
  size_t row_floats = (size_t)heads * work.max_chunks * (hd + 2);
  int microbatches = (prompt + m - 1) / m;
  double flops = 4.0 * heads * hd * ((double)prompt * (prompt + 1) / 2);
  double unique = (double)prompt * kv_dim * sizeof(float) * 2;
  Trace *trace = aligned_alloc(64, (sizeof(Trace) + 63) / 64 * 64);
  memset(trace, 0, sizeof(Trace));
  double times[MAX_REPS], last_times[MAX_REPS];
  double traced_seconds = 0;
  for (int pass = -1; pass <= reps; pass++) { /* -1: warm-up; 0..reps-1: timed; reps: traced */
    int traced = pass == reps;
    if (traced) {
      trace_reset(trace);
      k_attention_set_task_hook(on_task, trace);
    }
    double start = monotonic_seconds(), last = 0;
    for (int pos0 = 0; pos0 < prompt; pos0 += m) {
      int rows = prompt - pos0 < m ? prompt - pos0 : m;
      if (traced)
        atomic_fetch_add(&trace->call, 1);
      double t = monotonic_seconds();
      if (s->portable && rows > 1)
        k_attention_batch_portable(out, q, kc, vc, &work, heads, kv_heads, hd, seq, pos0, rows);
      else
        k_attention_batch(out, q, kc, vc, att, &work, heads, kv_heads, hd, seq, pos0, rows);
      last = monotonic_seconds() - t;
    }
    double elapsed = monotonic_seconds() - start;
    if (traced) {
      k_attention_set_task_hook(NULL, NULL);
      traced_seconds = elapsed;
    } else if (pass >= 0) {
      times[pass] = elapsed, last_times[pass] = last;
    }
  }
  TraceStats stats = trace_stats(trace, work.workers, traced_seconds);
  double sorted[MAX_REPS], sorted_last[MAX_REPS], sum = 0;
  memcpy(sorted, times, reps * sizeof(double)), memcpy(sorted_last, last_times, reps * sizeof(double));
  qsort(sorted, reps, sizeof(double), compare_double), qsort(sorted_last, reps, sizeof(double), compare_double);
  for (int i = 0; i < reps; i++)
    sum += times[i];
  double median = sorted[reps / 2], best = sorted[0];
  common_fields(s, "prefill", work.workers);
  printf("\"prompt\": %d, \"m\": %d, \"seq_len\": %d, \"microbatches\": %d, \"last_microbatch_rows\": %d, "
         "\"max_chunks\": %d, \"wave_rows_capacity\": %zu, \"reps\": %d, ",
         prompt, m, seq, microbatches, prompt - (microbatches - 1) * m, work.max_chunks, work.capacity / row_floats,
         reps);
  printf("\"ms_layer_best\": %.4f, \"ms_layer_median\": %.4f, \"ms_layer_mean\": %.4f, \"ms_layer_max\": %.4f, ",
         best * 1e3, median * 1e3, sum / reps * 1e3, sorted[reps - 1] * 1e3);
  json_list("ms_layer_all", times, reps, 1e3);
  printf(", \"last_microbatch_ms_median\": %.4f, \"last_microbatch_ms_best\": %.4f, ", sorted_last[reps / 2] * 1e3,
         sorted_last[0] * 1e3);
  printf("\"flops\": %.0f, \"gflops_best\": %.2f, \"gflops_median\": %.2f, \"unique_kv_bytes\": %.0f, ", flops,
         flops / best / 1e9, flops / median / 1e9, unique);
  /* One task streams K and V of its KV head: head_dim floats each per position. */
  json_stats(&stats, 2.0 * hd * sizeof(float));
  if (stats.valid)
    printf(", \"streamed_over_unique\": %.2f, \"flops_per_streamed_byte\": %.2f, \"traced_ms_layer\": %.4f",
           stats.span_positions * 2.0 * hd * sizeof(float) / unique,
           flops / (stats.span_positions * 2.0 * hd * sizeof(float)), traced_seconds * 1e3);
  printf("}\n");
  fflush(stdout);
  for (int i = 0; i < MAX_TRACKS; i++)
    free(trace->track[i].record);
  free(trace);
  k_attention_free(&work);
  region_release(&wr), region_release(&qr), region_release(&vr), region_release(&kr);
}

static void decode_cell(const Setup *s, int context, int reps, int cold_mib) {
  int heads = s->heads, kv_heads = s->kv_heads, hd = s->hd;
  size_t dim = (size_t)heads * hd, kv_dim = (size_t)kv_heads * hd, copy_floats = (size_t)context * kv_dim;
  double unique = (double)context * kv_dim * sizeof(float) * 2;
  int copies = (int)(((double)cold_mib * (1 << 20)) / unique);
  copies = copies < 1 ? 1 : copies > 64 ? 64 : copies;
  NumaRegion kr, vr, qr, wr;
  float *kc = carve(&kr, copies * copy_floats * sizeof(float), ROLE_KV_K, s->node);
  float *vc = carve(&vr, copies * copy_floats * sizeof(float), ROLE_KV_V, s->node);
  for (size_t i = 0; i < copy_floats; i++)
    kc[i] = uniform(), vc[i] = uniform();
  for (int c = 1; c < copies; c++)
    memcpy(kc + c * copy_floats, kc, copy_floats * sizeof(float)), memcpy(vc + c * copy_floats, vc, copy_floats * sizeof(float));
  region_mark_populated(&kr), region_mark_populated(&vr);
  float *q = carve(&qr, (2 * dim + (size_t)heads * context) * sizeof(float), ROLE_SCRATCH, s->node);
  float *out = q + dim, *att = out + dim;
  for (size_t i = 0; i < dim; i++)
    q[i] = uniform();
  region_mark_populated(&qr);
  AttentionWork work;
  float *memory = carve(&wr, k_attention_bytes(heads, kv_heads, hd, context), ROLE_WORK, s->node);
  k_attention_init_in(&work, heads, kv_heads, hd, context, memory);
  region_mark_populated(&wr);
  double flops = 4.0 * heads * hd * context;
  Trace *trace = aligned_alloc(64, (sizeof(Trace) + 63) / 64 * 64);
  memset(trace, 0, sizeof(Trace));
  double times[MAX_REPS], traced_seconds = 0;
  int traced_calls = reps < 16 ? reps : 16;
  for (int pass = -2; pass < reps + traced_calls; pass++) { /* -2, -1: warm-up; then timed; then traced */
    int traced = pass >= reps;
    if (pass == reps) {
      trace_reset(trace);
      k_attention_set_task_hook(on_task, trace);
    }
    int copy = (pass < 0 ? pass + 2 : pass) % copies;
    if (traced)
      atomic_fetch_add(&trace->call, 1);
    double t = monotonic_seconds();
    k_attention(out, q, kc + copy * copy_floats, vc + copy * copy_floats, att, &work, heads, kv_heads, hd, context,
                context - 1);
    double elapsed = monotonic_seconds() - t;
    if (traced)
      traced_seconds += elapsed;
    else if (pass >= 0)
      times[pass] = elapsed;
  }
  k_attention_set_task_hook(NULL, NULL);
  TraceStats stats = trace_stats(trace, work.workers, traced_seconds);
  double sorted[MAX_REPS], sum = 0;
  memcpy(sorted, times, reps * sizeof(double));
  qsort(sorted, reps, sizeof(double), compare_double);
  for (int i = 0; i < reps; i++)
    sum += times[i];
  double median = sorted[reps / 2], best = sorted[0];
  int chunks = k_attention_chunk_count(&work, kv_heads, context - 1);
  common_fields(s, "decode", work.workers);
  printf("\"context\": %d, \"chunks\": %d, \"kv_copies\": %d, \"reps\": %d, ", context, chunks, copies, reps);
  printf("\"ms_layer_best\": %.4f, \"ms_layer_median\": %.4f, \"ms_layer_mean\": %.4f, \"ms_layer_max\": %.4f, ",
         best * 1e3, median * 1e3, sum / reps * 1e3, sorted[reps - 1] * 1e3);
  json_list("ms_layer_all", times, reps, 1e3);
  printf(", \"flops\": %.0f, \"gflops_best\": %.2f, \"gflops_median\": %.2f, \"unique_kv_bytes\": %.0f, "
         "\"gbs_best\": %.2f, \"gbs_median\": %.2f, ",
         flops, flops / best / 1e9, flops / median / 1e9, unique, unique / best / 1e9, unique / median / 1e9);
  /* The trace covers several calls: scale its byte count back to one call. */
  TraceStats one = stats;
  one.span_positions /= traced_calls;
  json_stats(&one, 2.0 * hd * sizeof(float));
  printf("}\n");
  fflush(stdout);
  for (int i = 0; i < MAX_TRACKS; i++)
    free(trace->track[i].record);
  free(trace);
  k_attention_free(&work);
  region_release(&wr), region_release(&qr), region_release(&vr), region_release(&kr);
}

static void usage(void) {
  fprintf(stderr, "usage: attention-bench --mode prefill|decode [--shape 8b|1b|H/KV/HD,...] [--prompt LIST] [--m LIST] "
                  "[--context LIST] [--reps R] [--node N] [--kv-heads-local N] [--seq-extra N] [--cold-mib N] "
                  "[--portable]\n");
}

int main(int argc, char **argv) {
  const char *mode = "prefill", *shape_text = "8b";
  int node = -1, reps = 3, kv_local = 0, seq_extra = 256, cold_mib = 1024, portable = 0;
  int prompts[MAX_LIST] = {512, 2048, 8192}, np = 3, ms[MAX_LIST] = {16, 64, 256}, nm = 3;
  int contexts[MAX_LIST] = {8192, 32768}, nc = 2;
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    int more = i + 1 < argc;
    if (!strcmp(a, "--mode") && more)
      mode = argv[++i];
    else if (!strcmp(a, "--shape") && more)
      shape_text = argv[++i];
    else if (!strcmp(a, "--prompt") && more)
      np = parse_list(argv[++i], prompts, MAX_LIST);
    else if (!strcmp(a, "--m") && more)
      nm = parse_list(argv[++i], ms, MAX_LIST);
    else if (!strcmp(a, "--context") && more)
      nc = parse_list(argv[++i], contexts, MAX_LIST);
    else if (!strcmp(a, "--reps") && more)
      reps = atoi(argv[++i]);
    else if (!strcmp(a, "--node") && more)
      node = atoi(argv[++i]);
    else if (!strcmp(a, "--kv-heads-local") && more)
      kv_local = atoi(argv[++i]);
    else if (!strcmp(a, "--seq-extra") && more)
      seq_extra = atoi(argv[++i]);
    else if (!strcmp(a, "--cold-mib") && more)
      cold_mib = atoi(argv[++i]);
    else if (!strcmp(a, "--portable"))
      portable = 1;
    else {
      usage();
      return 2;
    }
  }
  int decode = !strcmp(mode, "decode");
  if ((!decode && strcmp(mode, "prefill")) || reps < 1 || reps > MAX_REPS || seq_extra < 0 || cold_mib < 0) {
    usage();
    return 2;
  }
  kernels_init();
  kernels_backend_init();
  char *shapes_copy = strdup(shape_text);
  for (char *token = strtok(shapes_copy, ","); token; token = strtok(NULL, ",")) {
    Setup setup = {.node = node, .portable = portable};
    if (parse_shape(token, &setup.shape)) {
      fprintf(stderr, "attention-bench: bad shape '%s'\n", token);
      return 2;
    }
    setup.heads = setup.shape.heads, setup.kv_heads = setup.shape.kv_heads, setup.hd = setup.shape.head_dim;
    if (kv_local) { /* one socket's share of a two-socket model: fewer KV heads, proportionally fewer query heads */
      if (kv_local < 1 || kv_local > setup.kv_heads || setup.heads * kv_local % setup.kv_heads) {
        fprintf(stderr, "attention-bench: --kv-heads-local %d does not divide the shape\n", kv_local);
        return 2;
      }
      setup.heads = setup.heads * kv_local / setup.kv_heads, setup.kv_heads = kv_local;
    }
    if (setup.heads / setup.kv_heads > 16 || setup.hd > 256) {
      fprintf(stderr, "attention-bench: the batched kernel needs at most 16 heads per KV head and head_dim <= 256\n");
      return 2;
    }
    if (decode)
      for (int c = 0; c < nc; c++)
        decode_cell(&setup, contexts[c], reps, cold_mib);
    else
      for (int p = 0; p < np; p++)
        for (int mi = 0; mi < nm; mi++)
          prefill_cell(&setup, prompts[p], ms[mi], reps, seq_extra);
  }
  free(shapes_copy);
  kernels_backend_free();
  return 0;
}
