#include "app.h"
#include "gemm.h"
#include "kernels.h"
#include "loader.h"
#include "numa.h"
#include "profile.h"
#include "workers.h"
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif
/* Matches CPU_SETSIZE, the most CPUs an affinity mask can name. */
#define CPU_LIMIT 1024
#ifndef BUILD_MODE
#define BUILD_MODE "custom"
#endif
#ifndef BUILD_FLAGS
#define BUILD_FLAGS "unknown"
#endif

/* Preserve valid UTF-8; encode invalid bytes as U+00xx for valid JSON. */
static void string(FILE *file, const char *s) {
  if (!s) {
    fputs("null", file);
    return;
  }
  fputc('"', file);
  const unsigned char *p = (const unsigned char *)s;
  while (*p) {
    if (*p == '"' || *p == '\\') {
      fputc('\\', file);
      fputc(*p++, file);
    } else if (*p < 32)
      fprintf(file, "\\u%04x", *p++);
    else if (*p < 127)
      fputc(*p++, file);
    else {
      int n = *p >= 0xc2 && *p <= 0xdf ? 2 : *p >= 0xe0 && *p <= 0xef ? 3 : *p >= 0xf0 && *p <= 0xf4 ? 4 : 0;
      int valid = n > 0;
      for (int i = 1; i < n && valid; i++)
        if (!p[i] || (p[i] & 0xc0) != 0x80)
          valid = 0;
      if (valid && ((*p == 0xe0 && p[1] < 0xa0) || (*p == 0xed && p[1] >= 0xa0) || (*p == 0xf0 && p[1] < 0x90) ||
                    (*p == 0xf4 && p[1] >= 0x90)))
        valid = 0;
      if (valid) {
        fwrite(p, 1, (size_t)n, file);
        p += n;
      } else
        fprintf(file, "\\u%04x", *p++);
    }
  }
  fputc('"', file);
}
static int same_file(const char *a, const char *b) {
  if (!a || !b)
    return 0;
  if (!strcmp(a, b))
    return 1;
  struct stat left, right;
  return stat(a, &left) == 0 && stat(b, &right) == 0 && left.st_dev == right.st_dev && left.st_ino == right.st_ino;
}
static void number(FILE *file, double value, int available) {
  if (available)
    fprintf(file, "%.9f", value);
  else
    fputs("null", file);
}
static void rate(FILE *file, size_t count, double seconds) {
  number(file, seconds > 0 ? (double)count / seconds : 0, count > 0 && seconds > 0);
}
static void timing(FILE *f, const char *key, double value, int available, int comma) {
  fprintf(f, "    \"%s\": ", key);
  number(f, value, available);
  fputs(comma ? ",\n" : "\n", f);
}
/* Per-stage phase totals; "other" is forward time outside every phase. */
static void profile_report(FILE *f) {
  if (!profile_enabled) {
    fputs("null", f);
    return;
  }
  const char *stages[STAGE_COUNT] = {"prefill", "decode"};
  const double forward[STAGE_COUNT] = {metrics.prefill_s, metrics.decode_s};
  const size_t passes[STAGE_COUNT] = {metrics.processed_tokens, metrics.decode_steps};
  fputs("{", f);
  for (int stage = 0; stage < STAGE_COUNT; stage++) {
    double phases = 0.0;
    fprintf(f, "%s\n    \"%s\": {\n      \"forward_passes\": %zu,\n      \"forward\": %.9g", stage ? "," : "",
            stages[stage], passes[stage], forward[stage]);
    for (int phase = 0; phase < PHASE_COUNT; phase++) {
      phases += profile.seconds[stage][phase];
      fprintf(f, ",\n      \"%s\": %.9g", profile_phase_name((Phase)phase), profile.seconds[stage][phase]);
    }
    fprintf(f, ",\n      \"other\": %.9g\n    }", forward[stage] - phases);
  }
  fputs("\n  }", f);
}

static void int_list(FILE *f, const int *values, int n) {
  fputc('[', f);
  for (int i = 0; i < n; i++)
    fprintf(f, "%s%d", i ? ", " : "", values[i]);
  fputc(']', f);
}

static void kv_fields(FILE *f, const KvStats *k) {
  fprintf(f, "\"reserved_bytes\": %zu, \"resident_bytes\": ", k->reserved_bytes);
  if (k->verified)
    fprintf(f, "%zu", k->resident_bytes);
  else
    fputs("null", f);
  fprintf(f, ", \"active_bytes\": %zu, \"node\": ", k->active_bytes);
  if (k->node >= 0)
    fprintf(f, "%d", k->node);
  else
    fputs("null", f);
  fprintf(f, ", \"verified\": %s, \"full\": %s, \"pages_queried\": %zu, \"pages_wrong_node\": %zu, "
             "\"pages_missing\": %zu",
          k->verified ? "true" : "false", k->full ? "true" : "false", k->pages_queried, k->pages_wrong_node,
          k->pages_missing);
}

/* The final state, with the snapshot taken when the prompt had been processed nested as after_prompt. */
static void kv_object(FILE *f) {
  if (!metrics.kv_final_ready) {
    fputs("null", f);
    return;
  }
  fputc('{', f);
  kv_fields(f, &metrics.kv_final);
  fputs(", \"after_prompt\": ", f);
  if (metrics.kv_prompt_ready) {
    fputc('{', f);
    kv_fields(f, &metrics.kv_prompt);
    fputc('}', f);
  } else
    fputs("null", f);
  fputc('}', f);
}

/* Schema v2: runtime, topology, teams, regions and placeholders for producers that do not exist yet (null). */
static void write_v2_sections(FILE *f) {
  const char *thp = region_thp_default() ? "on" : "off";
  fputs(",\n  \"runtime\": {\n    \"backend\": ", f);
  string(f, kernels_backend_name());
  fputs(",\n    \"sockets\": 1,\n    \"placement\": ", f);
  string(f, placement_active());
  fputs(",\n    \"thp\": ", f);
  string(f, thp);
  fputs(",\n    \"precision_mode\": ", f);
  string(f, !strcmp(engine_precision(), "bf16_from_fp32") ? "bf16" : engine_precision());
  fputs(",\n    \"prefill_batch\": ", f);
  if (metrics.runtime_ready)
    fprintf(f, "%d", engine_max_batch(&model));
  else
    fputs("null", f);
  fputs(",\n    \"prefill_logits\": ", f);
  string(f, metrics.runtime_ready ? (model.prefill_logits == LOGITS_ALL ? "all" : "last") : NULL);
  fputs(",\n    \"kernels_prefill\": ", f);
  string(f, metrics.runtime_ready && model.prefill_gemm ? model.prefill_kernel : NULL);
  fputs("\n  }", f);

  const Topology *topology;
  char error[512];
  fputs(",\n  \"topology\": ", f);
  if (numa_topology_try(&topology, error, sizeof(error)) == 0) {
    fputs("{\n    \"sockets\": [", f);
    for (int s = 0; s < topology->nsockets; s++) {
      const SocketTopo *k = &topology->socket[s];
      fprintf(f, "%s\n      {\"package\": %d, \"cpus\": ", s ? "," : "", k->package);
      if (k->cpus)
        int_list(f, k->cpus, k->ncpus);
      else
        fprintf(f, "null, \"threads\": %d", k->ncpus);
      fprintf(f, ", \"ddr_node\": %d, \"hbm_node\": %d, \"ddr_allowed\": %s, \"hbm_allowed\": %s, \"classification\": ",
              k->ddr_node, k->hbm_node, k->ddr_allowed ? "true" : "false", k->hbm_allowed ? "true" : "false");
      string(f, numa_class_name(topology->classification));
      fputc('}', f);
    }
    fputs("\n    ],\n    \"mems_allowed\": [", f);
    int first = 1;
    for (int n = 0; n <= topology->max_node; n++)
      if (topology->mems_allowed[n / (8 * sizeof(unsigned long))] >> (n % (8 * sizeof(unsigned long))) & 1) {
        fprintf(f, "%s%d", first ? "" : ", ", n);
        first = 0;
      }
    fputs("],\n    \"mems_allowed_list\": ", f);
    string(f, topology->classification == CLASS_VIRTUAL ? NULL : topology->mems_allowed_list);
    fputs(",\n    \"cpus_allowed\": ", f);
    int affinity[CPU_LIMIT];
    int count = numa_affinity_cpus(affinity, CPU_LIMIT);
    if (count >= 0)
      int_list(f, affinity, count);
    else
      fputs("null", f);
    fputs("\n  },\n  \"topology_error\": null", f);
  } else {
    fputs("null,\n  \"topology_error\": ", f);
    string(f, error);
  }

  fputs(",\n  \"teams\": ", f);
  if (pool_summary_count() == 0)
    fputs("null", f);
  else {
    fputc('[', f);
    for (int t = 0; t < pool_summary_count(); t++) {
      const TeamSummary *team = pool_summary(t);
      fprintf(f, "%s\n    {\"index\": %d, \"threads\": %d, \"cpus\": ", t ? "," : "", team->index, team->threads);
      if (team->pinned)
        int_list(f, team->assigned, team->threads);
      else
        fputs("null", f);
      fprintf(f, ", \"affinity_verified\": %s}", team->verified ? "true" : "false");
    }
    fputs("\n  ]", f);
  }

  fputs(",\n  \"regions\": [", f);
  for (int i = 0; i < region_history_count(); i++) {
    const RegionRecord *r = region_history_at(i);
    fprintf(f, "%s\n    {\"role\": \"%s\", \"socket\": %d, \"node\": ", i ? "," : "", region_role_name(r->role),
            r->socket);
    if (r->node >= 0)
      fprintf(f, "%d", r->node);
    else
      fputs("null", f);
    fprintf(f, ", \"bytes\": %zu, \"thp_requested\": %s, ", r->bytes, r->thp_requested ? "true" : "false");
    if (r->checked)
      fprintf(f,
              "\"thp_bytes\": %zu, \"verified\": \"%s\", \"pages_queried\": %zu, \"pages_mismatched\": %zu, "
              "\"pages_missing\": %zu}",
              r->thp_bytes, r->full ? "full" : "sampled", r->pages_queried, r->pages_mismatched, r->pages_missing);
    else
      fputs("\"thp_bytes\": null, \"verified\": \"unverified\", \"pages_queried\": null, \"pages_mismatched\": "
            "null, \"pages_missing\": null}",
            f);
  }
  fputs(region_history_count() ? "\n  ]" : "]", f);
  fputs(",\n  \"kv\": ", f);
  kv_object(f);
  fprintf(f, ",\n  \"phase_seconds\": null,\n  \"prefill\": {\"microbatches\": %zu, \"tokens\": %zu, \"seconds\": %.9f, \"logits_rows\": %zu}",
          metrics.prefill_microbatches, metrics.processed_tokens, metrics.prefill_s, metrics.prefill_logits_rows);
}

void write_report(void) {
  if (!options.json)
    return;
  const char *inputs[] = {options.checkpoint, options.tokenizer, options.prompt_file};
  for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++)
    if (same_file(options.json, inputs[i])) {
      log_message(LOG_ERROR, "report path '%s' aliases input '%s'", options.json, inputs[i]);
      _Exit(EXIT_FAILURE);
    }
  FILE *f = fopen(options.json, "wb");
  if (!f) {
    log_message(LOG_ERROR, "cannot create report '%s': %s", options.json, strerror(errno));
    _Exit(EXIT_FAILURE);
  }
  struct utsname system;
  int system_ok = uname(&system) == 0;
  time_t now = time(NULL);
  struct tm utc;
  char timestamp[32] = "";
  if (gmtime_r(&now, &utc))
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%SZ", &utc);
  fputs("{\n  \"schema_version\": 2,\n  \"timestamp_utc\": ", f);
  string(f, timestamp);
  fputs(",\n  \"status\": ", f);
  string(f, metrics.completed ? "completed" : "error");
  fputs(",\n  \"error\": ", f);
  string(f, failure_message[0] ? failure_message : NULL);
  fputs(",\n  \"stop_reason\": ", f);
  string(f, metrics.stop_reason);
  fputs(",\n  \"model\": {\n    \"identifier\": ", f);
  string(f, options.checkpoint);
  fputs(",\n    \"checkpoint_path\": ", f);
  string(f, options.checkpoint);
  fputs(",\n    \"precision\": ", f);
  string(f, engine_precision());
  fputs(",\n    \"checkpoint_format\": ", f);
  if (metrics.checkpoint_ready)
    fprintf(f, "%d", model.config.format);
  else
    fputs("null", f);
  fprintf(f, ",\n    \"quantization_group_size\": %d,\n    \"configuration\": ", engine_group_size());
  if (metrics.checkpoint_ready) {
    Config *p = &model.config;
    fprintf(f,
            "{\"dim\": %d, \"hidden_dim\": %d, \"n_layers\": %d, \"n_heads\": %d, \"n_kv_heads\": %d, \"vocab_size\": "
            "%d, \"seq_len\": %d, \"max_position_embeddings\": %d, \"head_dim\": %d, \"rope_theta\": %.9g, "
            "\"rms_norm_epsilon\": %.9g, \"shared_embeddings\": %s, \"rope_type\": \"%s\", "
            "\"rope_factor\": %.9g, \"rope_low_freq_factor\": %.9g, \"rope_high_freq_factor\": %.9g, "
            "\"rope_original_context\": %d, \"bos_id\": %d, \"eos_id\": %d, \"eot_id\": %d, \"eom_id\": %d}",
            p->dim, p->hidden_dim, p->n_layers, p->n_heads, p->n_kv_heads, p->vocab_size, p->seq_len, p->max_seq_len,
            p->head_dim, p->rope_theta, p->norm_eps, p->shared ? "true" : "false", p->rope_type ? "llama3" : "default",
            p->rope_factor, p->rope_low, p->rope_high, p->rope_original, p->bos_id, p->eos_id, p->eot_id, p->eom_id);
    fputs(",\n    \"name\": ", f);
    string(f, p->name[0] ? p->name : NULL);
    /* Counts refer to stored model parameters, excluding legacy RoPE tables and
     * quantization scales; a tied classifier is counted once. */
    uint64_t d = p->dim, h = p->hidden_dim, l = p->n_layers, kv = (uint64_t)p->head_dim * p->n_kv_heads;
    uint64_t count =
        (uint64_t)p->vocab_size * d * (p->shared ? 1 : 2) + (2 * l + 1) * d + l * (2 * d * d + 2 * d * kv + 3 * d * h);
    fprintf(f, ",\n    \"parameter_count\": %llu", (unsigned long long)count);
  } else
    fputs("null", f);
  fputs("\n  },\n  \"system\": {\n    \"os\": ", f);
  string(f, system_ok ? system.sysname : NULL);
  fputs(",\n    \"hostname\": ", f);
  string(f, system_ok ? system.nodename : NULL);
  fputs(",\n    \"kernel\": ", f);
  string(f, system_ok ? system.release : NULL);
  fputs(",\n    \"architecture\": ", f);
  string(f, system_ok ? system.machine : NULL);
  long cpus = sysconf(_SC_NPROCESSORS_ONLN);
  fputs(",\n    \"online_cpus\": ", f);
  if (cpus > 0)
    fprintf(f, "%ld", cpus);
  else
    fputs("null", f);
  fputs(",\n    \"affinity_cpus\": ", f);
  int affinity[CPU_LIMIT];
  int affinity_count = numa_affinity_cpus(affinity, CPU_LIMIT);
  if (affinity_count >= 0) {
    fputc('[', f);
    for (int i = 0; i < affinity_count; i++)
      fprintf(f, "%s%d", i ? ", " : "", affinity[i]);
    fputc(']', f);
  } else
    fputs("null", f);
  fputs(",\n    \"numa_online_nodes\": ", f);
  char topology[256];
  string(f, numa_online_nodes(topology, sizeof(topology)) == 0 ? topology : NULL);
  fputs("\n  },\n  \"build\": {\n    \"compiler\": ", f);
  string(f, __VERSION__);
  fputs(",\n    \"mode\": ", f);
  string(f, BUILD_MODE);
  fputs(",\n    \"flags\": ", f);
  string(f, BUILD_FLAGS);
#ifdef _OPENMP
  fprintf(f, ",\n    \"openmp\": true,\n    \"openmp_version\": %d", _OPENMP);
#else
  fputs(",\n    \"openmp\": false,\n    \"openmp_version\": null", f);
#endif
  fputs(",\n    \"kernels\": ", f);
  if (kernels_name())
    string(f, kernels_name());
  else
    fputs("null", f);
  fputs(",\n    \"amx_permission\": ", f);
  string(f, gemm_amx_permission());
  fputs(",\n    \"openmp_max_threads\": ", f);
#ifdef _OPENMP
  fprintf(f, "%d", omp_get_max_threads());
#else
  fputs("1", f);
#endif
  fputs("\n  },\n  \"parameters\": {\n    \"mode\": ", f);
  string(f, options.mode);
  fprintf(f, ",\n    \"chat_template\": %s", options.chat_template ? "true" : "false");
  fprintf(f,
          ",\n    \"benchmark\": %s,\n    \"seed\": %llu,\n    \"temperature\": %.9g,\n    \"top_p\": %.9g,\n    "
          "\"max_forward_passes\": %d,\n    \"generation_length\": ",
          options.benchmark ? "true" : "false", options.seed, options.temperature, options.topp, options.steps);
  if (options.generation_length >= 0)
    fprintf(f, "%d", options.generation_length);
  else
    fputs("null", f);
  fprintf(f, ",\n    \"text_suppressed\": %s,\n    \"tokenizer_path\": ", options.no_text ? "true" : "false");
  string(f, options.tokenizer);
  fputs(",\n    \"prompt\": ", f);
  string(f, options.prompt);
  fputs(",\n    \"prompt_file\": ", f);
  string(f, options.prompt_file);
  fputs(",\n    \"system_prompt\": ", f);
  string(f, options.system_prompt);
  fputs(",\n    \"prefill_batch\": ", f);
  if (metrics.runtime_ready)
    fprintf(f, "%d", engine_max_batch(&model));
  else
    fputs("null", f);
  fputs(",\n    \"prefill_logits\": ", f);
  string(f, metrics.runtime_ready ? (model.prefill_logits == LOGITS_ALL ? "all" : "last") : NULL);
  fprintf(f,
          "\n  },\n  \"tokens\": {\n    \"input_prompt\": %zu,\n    \"processed_prompt\": %zu,\n    "
          "\"generated_output\": %zu,\n    \"forward_passes\": %zu,\n    \"decode_steps\": %zu,\n    \"sampled\": "
          "%zu,\n    \"sampled_stop\": %zu\n  },\n  \"timing_seconds\": {\n",
          metrics.input_tokens, metrics.processed_tokens, metrics.output_tokens, metrics.forward_passes,
          metrics.decode_steps, metrics.sampled_tokens, metrics.stop_tokens);
  timing(f, "checkpoint_load", metrics.checkpoint_s, metrics.checkpoint_ready, 1);
  timing(f, "tokenizer_init", metrics.tokenizer_s, metrics.tokenizer_ready, 1);
  timing(f, "runtime_init", metrics.runtime_s, metrics.runtime_ready, 1);
  timing(f, "prompt_tokenization", metrics.tokenization_s, metrics.turns > 0 && metrics.completed, 1);
  timing(f, "prefill", metrics.prefill_s, metrics.processed_tokens > 0, 1);
  timing(f, "decode", metrics.decode_s, metrics.decode_steps > 0, 1);
  timing(f, "total_inference", metrics.inference_s, metrics.turns > 0 && metrics.completed, 1);
  timing(f, "end_to_end", metrics.end_to_end_s, 1, 1);
  timing(f, "time_to_first_token", metrics.ttft_s, metrics.has_first_token, 1);
  timing(f, "average_time_per_generated_token", metrics.output_tokens ? metrics.inference_s / metrics.output_tokens : 0,
         metrics.output_tokens > 0, 1);
  timing(f, "average_decode_step", metrics.decode_steps ? metrics.decode_s / metrics.decode_steps : 0,
         metrics.decode_steps > 0, 1);
  /* weight_pack is the BF16 packing of C5; the placement timings come from the owned loader (A4). */
  timing(f, "weight_pack", metrics.weight_pack_s, metrics.weight_packed, 1);
  timing(f, "placement_populate", metrics.placement_populate_s, metrics.placement_owned, 1);
  timing(f, "placement_verify", metrics.placement_verify_s, metrics.placement_owned, 0);
  fputs("  },\n  \"throughput_tokens_per_second\": {\n    \"prefill\": ", f);
  rate(f, metrics.processed_tokens, metrics.prefill_s);
  fputs(",\n    \"decode\": ", f);
  rate(f, metrics.decode_steps, metrics.decode_s);
  fputs(",\n    \"overall_forward\": ", f);
  rate(f, metrics.forward_passes, metrics.inference_s);
  fputs(",\n    \"output\": ", f);
  rate(f, metrics.output_tokens, metrics.inference_s);
  fputs("\n  },\n  \"profile_seconds\": ", f);
  profile_report(f);
  write_v2_sections(f);
  fputs(",\n  \"memory_bytes\": {\n    \"model_file\": ", f);
  if (metrics.checkpoint_ready)
    fprintf(f, "%zu", model.file_bytes);
  else
    fputs("null", f);
  fputs(",\n    \"mapped_checkpoint\": ", f);
  if (metrics.checkpoint_ready)
    fprintf(f, "%zu", model.mapped_bytes);
  else
    fputs("null", f);
  const char *names[] = {
      "kv_cache_allocated_peak",       "activations_allocated_peak", "dequantized_embedding_allocated_peak",
      "model_metadata_allocated_peak", "tokenizer_allocated_peak",   "sampler_allocated_peak",
      "prompt_buffers_allocated_peak", "owned_weights_peak",         "packed_weights_peak"};
  for (int i = 0; i < MEM_COUNT; i++)
    fprintf(f, ",\n    \"%s\": %zu", names[i], memory_peak[i]);
  fprintf(f, ",\n    \"explicit_allocations_peak\": %zu,\n    \"peak_resident\": ", memory_total_peak);
  struct rusage resource;
  if (getrusage(RUSAGE_SELF, &resource) == 0) {
#ifdef __APPLE__
    fprintf(f, "%llu", (unsigned long long)resource.ru_maxrss);
#else
    fprintf(f, "%llu", (unsigned long long)resource.ru_maxrss * 1024);
#endif
  } else
    fputs("null", f);
  fputs("\n  }\n}\n", f);
  int write_error = ferror(f);
  if (fclose(f))
    write_error = 1;
  if (write_error) {
    log_message(LOG_ERROR, "cannot write report '%s': %s", options.json, strerror(errno));
    _Exit(EXIT_FAILURE);
  }
}
