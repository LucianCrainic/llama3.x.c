#include "profile.h"
#include "runtime.h"
#include <stdlib.h>
#include <string.h>

int profile_enabled;
Stage profile_stage;
Profile profile;

void profile_init(void) {
  const char *value = getenv("LLAMA_PROFILE");
  if (value && *value && strcmp(value, "0") && strcmp(value, "1"))
    fail("LLAMA_PROFILE must be 0 or 1");
  profile_enabled = value && !strcmp(value, "1");
  memset(&profile, 0, sizeof(profile));
}

const char *profile_phase_name(Phase phase) {
  static const char *names[PHASE_COUNT] = {"embedding", "rope_table", "rmsnorm", "quantize",     "qkv", "rope",
                                           "kv_store",  "attention",  "wo",      "w1_w3_swiglu", "w2",  "classifier"};
  return names[phase];
}

double profile_start(void) {
  return profile_enabled ? monotonic_seconds() : 0.0;
}

void profile_stop(Phase phase, double start) {
  if (!profile_enabled)
    return;
  profile.seconds[profile_stage][phase] += monotonic_seconds() - start;
  profile.calls[profile_stage][phase]++;
}
