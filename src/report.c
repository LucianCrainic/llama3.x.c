#include "app.h"
#include "kernels.h"
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
#ifdef __linux__
#include <sched.h>
#endif
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
  fputs("{\n  \"schema_version\": 1,\n  \"timestamp_utc\": ", f);
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
#ifdef __linux__
  cpu_set_t affinity;
  if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0) {
    fputc('[', f);
    int comma = 0;
    for (int i = 0; i < CPU_SETSIZE; i++)
      if (CPU_ISSET(i, &affinity)) {
        fprintf(f, "%s%d", comma ? ", " : "", i);
        comma = 1;
      }
    fputc(']', f);
  } else
    fputs("null", f);
#else
  fputs("null", f);
#endif
  fputs(",\n    \"numa_online_nodes\": ", f);
#ifdef __linux__
  FILE *nodes = fopen("/sys/devices/system/node/online", "r");
  char topology[256];
  if (nodes && fgets(topology, sizeof(topology), nodes)) {
    topology[strcspn(topology, "\r\n")] = 0;
    string(f, topology);
  } else
    fputs("null", f);
  if (nodes)
    fclose(nodes);
#else
  fputs("null", f);
#endif
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
         metrics.decode_steps > 0, 0);
  fputs("  },\n  \"throughput_tokens_per_second\": {\n    \"prefill\": ", f);
  rate(f, metrics.processed_tokens, metrics.prefill_s);
  fputs(",\n    \"decode\": ", f);
  rate(f, metrics.decode_steps, metrics.decode_s);
  fputs(",\n    \"overall_forward\": ", f);
  rate(f, metrics.forward_passes, metrics.inference_s);
  fputs(",\n    \"output\": ", f);
  rate(f, metrics.output_tokens, metrics.inference_s);
  fputs("\n  },\n  \"memory_bytes\": {\n    \"model_file\": ", f);
  if (metrics.checkpoint_ready)
    fprintf(f, "%zu", model.mapped_bytes);
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
      "prompt_buffers_allocated_peak"};
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
