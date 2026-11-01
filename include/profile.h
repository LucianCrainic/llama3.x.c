/* Opt-in wall-clock profile of the forward pass, enabled by LLAMA_PROFILE=1.
 * Every phase is serial or ends at an OpenMP barrier, so the calling thread's
 * wall time is the phase's time. Disabled, each PROFILE costs one branch. */
#ifndef PROFILE_H
#define PROFILE_H

#include <stddef.h>

typedef enum {
  PHASE_EMBED,
  PHASE_ROPE_TABLE,
  PHASE_RMSNORM,
  PHASE_QUANTIZE,
  PHASE_QKV,
  PHASE_ROPE,
  PHASE_KV_STORE,
  PHASE_ATTENTION,
  PHASE_WO,
  PHASE_W1_W3,
  PHASE_W2,
  PHASE_CLASSIFIER,
  PHASE_COUNT
} Phase;
typedef enum { STAGE_PREFILL, STAGE_DECODE, STAGE_COUNT } Stage;

typedef struct {
  double seconds[STAGE_COUNT][PHASE_COUNT];
  size_t calls[STAGE_COUNT][PHASE_COUNT];
} Profile;

extern int profile_enabled;
extern Stage profile_stage;
extern Profile profile;

void profile_init(void); // reads LLAMA_PROFILE=0|1
const char *profile_phase_name(Phase phase);
double profile_start(void);
void profile_stop(Phase phase, double start);

#define PROFILE(phase, ...)                                                                                            \
  do {                                                                                                                 \
    double profile_start_ = profile_start();                                                                           \
    __VA_ARGS__;                                                                                                       \
    profile_stop(phase, profile_start_);                                                                               \
  } while (0)

#endif
