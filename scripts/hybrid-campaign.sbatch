#!/usr/bin/env bash
#SBATCH --job-name=hybrid-campaign
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=06:00:00

# P4 baseline campaign of the HBM/DDR hybrid study (docs/hybrid-memory-results.md): decode at a ladder of contexts and TTFT, per
# placement and socket count, with the uncore CAS counters over the decode window, the phase profile, the placement proof and the
# token trace of every run. Submit to one Flat node with both HBM nodes allowed; every cell of one job shares the node, so the
# cells of a step form A/B pairs. Built on scripts/concurrent-tiers.sbatch (cells, interleaved repeats, perf wrapper, run checks)
# and scripts/hbm-ddr-research.sbatch (counter events, synthetic K/V build): the two scripts hold their helpers inline and cannot
# be sourced, so the few lines this script needs of them are repeated here rather than changed there; tools/hbm_ddr_results.py,
# tools/check_placement.py and tools/make_prompts.py are used as they are.
#
# Required: ROOT (staged tree), OUT (new directory), PYTHON, STEPS (words of the steps below), and MODEL_8B / MODEL_70B (directories
# holding int8-v3.bin and tokenizer-v2.bin) for the steps that load that model. Optional: REPEATS=3 (interleaved repetitions per
# cell), REPEATS_AB=5 (the steps whose cells form an A/B pair: c3 c4 c6 c7), ONLY_MEAS (words: measurement names to keep, or step:name to keep one
# step's, for a repeat), PLAN_ONLY=1 (build, admit every cell with --plan-only, write the cell list, stop), DRY_RUN=1 (print the cells, nothing
# else). List variables hold spaces: set them in the environment of the sbatch call, never inside --export, which splits its
# argument on commas. Nothing here names a host, a path or a credential: locations arrive through the environment.
#
# The engine takes one --warm-context per process: a "context ladder" is one process per (cell, context, repetition), each
# loading the model again. Contexts above 32768 on the 70B fill the K/V synthetically (a SYNTHETIC_KV=1 build, 1024 real primer
# positions); a measurement named d<N>s is synthetic, d<N> real, and d32768s runs next to d32768 as the check.
#
# Steps (every cell: LLAMA_GEMV_STAGGER=1, LLAMA_PLACEMENT_STRICT=1, explicit topology, engine defaults otherwise):
#   c0-8b-1s, c0-8b-2s  8B: hbm-only, ddr, serial hybrid with n weight and K/V layers in HBM (n = 32 24 16 8 0), concurrent rows
#                       (DDR fractions 0.1 0.2 0.3, 14 DDR cores); decode at 4096 and 32768, TTFT at 2048
#   c1                  70B 2S, hybrid 80/80 with the packed copy in DDR (hbm-only overflows with it): decode 4096, 32768,
#                       65536 (synthetic), TTFT 2048 and 8192, the 32768 synthetic check
#   c2                  70B 2S, ddr: decode 4096, 32768, 131072 (synthetic), TTFT 2048, the 32768 synthetic check
#   c3                  70B 2S: serial hybrid against concurrent rows at 32768 (real) and 65536 (synthetic)
#   c4                  70B 2S: serial hybrid (weights in HBM, K/V by the planner) against concurrent rows at 65536, 98304 and
#                       131072 (synthetic); a cell the planner does not admit is recorded in skipped.tsv
#   c6                  70B 1S: serial hybrid, concurrent rows and ddr at 32768 (synthetic, real fills of the serial cell as check)
#   c7                  70B 2S: AMX packed prefill (hybrid 80/80, packed in DDR) against LLAMA_KERNELS=avx512, TTFT 2048 and 8192
set -euo pipefail

: "${ROOT:?}" "${OUT:?}" "${PYTHON:?}" "${STEPS:?}"
repeats=${REPEATS:-3}
repeats_ab=${REPEATS_AB:-5}
prime_prompt="Print a long numbered list of prime numbers. Keep listing primes for at least 1000 tokens. Start at 2."
syn_prime=1024
ttft_generation=32
ctx_limit=131072
[[ ! -e $OUT ]] || { echo "OUT '$OUT' already exists" >&2; exit 2; }
[[ ${DRY_RUN:-0} == 1 || -n ${SLURM_JOB_ID:-} ]] || { echo "submit through Slurm" >&2; exit 2; }

base=(LLAMA_BACKEND=teams LLAMA_THREADS=56 LLAMA_PLACEMENT_STRICT=1 LLAMA_VERIFY_SAMPLES=16384 LLAMA_KV_VERIFY=1 LLAMA_GEMV_STAGGER=1
  LLAMA_TOPOLOGY=explicit:0/2/0,1/3/1)

# ---- the steps ----------------------------------------------------------------------------------------------------------------
# select_step STEP sets: model sockets layers checkpoint tokenizer generation reps_default cfg bind needs_syn and the parallel arrays
# set_names/set_envs (settings) and m_name m_kind m_warm m_ctx m_gen m_syn m_prompt m_reps m_sets (measurements; m_sets is the
# space-separated settings the measurement runs).
set_names=(); set_envs=()
m_name=(); m_kind=(); m_warm=(); m_ctx=(); m_gen=(); m_syn=(); m_prompt=(); m_reps=(); m_sets=()

add_set() { set_names+=("$1"); set_envs+=("$2"); }

ctx_for() { local c=$(($1 + 256)); echo $((c > ctx_limit ? ctx_limit : c)); }

# add_decode NAME WARM SYN "SETTINGS" [REPS]: decode at WARM warm positions (real or synthetic K/V)
add_decode() {
  local warm=$(($1 > ctx_limit - 256 ? ctx_limit - 256 : $1)) # the context ends at the model's limit: nominal 131072 warms 130816
  m_name+=("$2"); m_kind+=(decode); m_warm+=("$warm"); m_ctx+=("$(ctx_for "$warm")"); m_gen+=("$generation"); m_syn+=("$3")
  m_prompt+=(prime); m_reps+=("${5:-$reps_default}"); m_sets+=("$4")
}
# add_ttft NAME PROMPT_TOKENS "SETTINGS": prefill of a prompt file of that many tokens (chat template included), no warm context
add_ttft() { # NAME PROMPT_TOKENS "SETTINGS" [REPS]
  m_name+=("$1"); m_kind+=(ttft); m_warm+=(0); m_ctx+=("$(ctx_for "$2")"); m_gen+=("$ttft_generation"); m_syn+=(0)
  m_prompt+=("$2"); m_reps+=("${4:-$reps_default}"); m_sets+=("$3")
}

select_step() {
  local step=$1 n
  set_names=(); set_envs=(); m_name=(); m_kind=(); m_warm=(); m_ctx=(); m_gen=(); m_syn=(); m_prompt=(); m_reps=(); m_sets=()
  needs_syn=0
  case $step in
    c0-8b-1s | c0-8b-2s) model=8b ;;
    c1 | c2 | c3 | c4 | c6 | c7) model=70b ;;
    *) echo "unknown step '$step'" >&2; exit 2 ;;
  esac
  case $step in c0-8b-1s | c6) sockets=1 ;; *) sockets=2 ;; esac
  if [[ $model == 8b ]]; then
    : "${MODEL_8B:?MODEL_8B is required for $step}"
    checkpoint=$MODEL_8B/int8-v3.bin; tokenizer=$MODEL_8B/tokenizer-v2.bin; layers=32; generation=256
  else
    : "${MODEL_70B:?MODEL_70B is required for $step}"
    checkpoint=$MODEL_70B/int8-v3.bin; tokenizer=$MODEL_70B/tokenizer-v2.bin; layers=80; generation=96
  fi
  reps_default=$repeats
  case $step in c3 | c4 | c6 | c7) reps_default=$repeats_ab ;; esac
  cfg=("${base[@]}" "LLAMA_SOCKETS=$sockets")
  bind=()
  [[ $sockets == 2 ]] || bind=(numactl --cpunodebind=0)
  local hyb="LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_CLASSIFIER=hbm LLAMA_HYBRID_PACKED=ddr"
  local rows="$hyb LLAMA_HYBRID_SPLIT=rows"
  case $step in
    c0-8b-1s | c0-8b-2s)
      add_set hbm-only "LLAMA_PLACEMENT=hbm-only"
      add_set ddr "LLAMA_PLACEMENT=ddr"
      for n in 32 24 16 8 0; do add_set "hyb-$n" "$hyb LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=$n LLAMA_HYBRID_KV_HBM_LAYERS=$n"; done
      for n in 0.1 0.2 0.3; do
        add_set "rows-r$n-c14" "$rows LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_DDR_FRACTION=$n LLAMA_HYBRID_DDR_CORES=14"
      done
      local all="${set_names[*]}"
      add_decode 4096 d4096 0 "$all"
      add_decode 32768 d32768 0 "$all"
      add_ttft t2048 2048 "$all"
      ;;
    c1)
      add_set hyb80 "$hyb LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=80 LLAMA_HYBRID_KV_HBM_LAYERS=80"
      add_decode 4096 d4096 0 hyb80
      add_decode 32768 d32768 0 hyb80
      add_decode 32768 d32768s 1 hyb80
      add_decode 65536 d65536s 1 hyb80
      add_ttft t2048 2048 hyb80 5 # 5: a TTFT run that overlaps the startup pause is dropped (tools/hybrid_campaign_summary.py)
      add_ttft t8192 8192 hyb80
      needs_syn=1
      ;;
    c2)
      add_set ddr "LLAMA_PLACEMENT=ddr"
      add_decode 4096 d4096 0 ddr
      add_decode 32768 d32768 0 ddr
      add_decode 32768 d32768s 1 ddr
      add_decode 131072 d131072s 1 ddr
      add_ttft t2048 2048 ddr 5
      needs_syn=1
      ;;
    c3)
      add_set serial "$hyb"
      add_set serial-kv80 "$hyb LLAMA_HYBRID_KV_HBM_LAYERS=$layers"
      add_set rows-f0.1-c8 "$rows LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_DDR_FRACTION=0.1 LLAMA_HYBRID_DDR_CORES=8"
      add_set rows-f0.2-c14 "$rows LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_DDR_FRACTION=0.2 LLAMA_HYBRID_DDR_CORES=14"
      add_set rows-auto-c14 "$rows LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_DDR_CORES=14"
      add_set rows-auto-c20 "$rows LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_DDR_CORES=20"
      add_decode 32768 d32768 0 "serial rows-f0.1-c8 rows-f0.2-c14"
      add_decode 65536 d65536s 1 "serial serial-kv80 rows-auto-c14 rows-auto-c20"
      add_decode 32768 d32768s 1 serial "$repeats" # the synthetic check of the serial cell
      needs_syn=1
      ;;
    c4)
      # serial: every weight layer in HBM, the K/V layers the planner leaves room for. rows-auto: every K/V layer in HBM, the
      # weights spill by the smallest fraction that fits (not admitted where the K/V alone overflows HBM). rows-kvmatch: the
      # K/V layers of the serial cell, the fraction left to the planner (@KVMATCH@ is the serial plan's kv_hbm_layers).
      add_set serial-w80 "$hyb LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=80"
      add_set rows-auto-c20 "$rows LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_DDR_CORES=20"
      add_set rows-kvmatch-c20 "$rows LLAMA_HYBRID_KV_HBM_LAYERS=@KVMATCH@ LLAMA_HYBRID_DDR_CORES=20"
      add_decode 65536 d65536s 1 "serial-w80 rows-auto-c20 rows-kvmatch-c20"
      add_decode 98304 d98304s 1 "serial-w80 rows-auto-c20 rows-kvmatch-c20"
      add_decode 131072 d131072s 1 "serial-w80 rows-auto-c20 rows-kvmatch-c20"
      needs_syn=1
      ;;
    c6)
      add_set serial "$hyb"
      add_set serial-kv80 "$hyb LLAMA_HYBRID_KV_HBM_LAYERS=$layers"
      add_set rows-auto-c20 "$rows LLAMA_HYBRID_KV_HBM_LAYERS=$layers LLAMA_HYBRID_DDR_CORES=20"
      add_set ddr "LLAMA_PLACEMENT=ddr"
      add_decode 32768 d32768s 1 "serial serial-kv80 rows-auto-c20 ddr"
      add_decode 32768 d32768 0 serial "$repeats"
      needs_syn=1
      ;;
    c7)
      add_set amx "$hyb LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=80 LLAMA_HYBRID_KV_HBM_LAYERS=80"
      add_set avx512 "LLAMA_PLACEMENT=hybrid LLAMA_HYBRID_CLASSIFIER=hbm LLAMA_HYBRID_WEIGHTS_HBM_LAYERS=80 LLAMA_HYBRID_KV_HBM_LAYERS=80 LLAMA_KERNELS=avx512"
      add_ttft t2048 2048 "amx avx512"
      add_ttft t8192 8192 "amx avx512"
      ;;
  esac
  if [[ -n ${ONLY_MEAS:-} ]]; then # keep the measurements named in ONLY_MEAS
    local k keep_name=() keep_kind=() keep_warm=() keep_ctx=() keep_gen=() keep_syn=() keep_prompt=() keep_reps=() keep_sets=()
    for ((k = 0; k < ${#m_name[@]}; k++)); do
      [[ " $ONLY_MEAS " == *" ${m_name[k]} "* || " $ONLY_MEAS " == *" $step:${m_name[k]} "* ]] || continue
      keep_name+=("${m_name[k]}"); keep_kind+=("${m_kind[k]}"); keep_warm+=("${m_warm[k]}"); keep_ctx+=("${m_ctx[k]}")
      keep_gen+=("${m_gen[k]}"); keep_syn+=("${m_syn[k]}"); keep_prompt+=("${m_prompt[k]}"); keep_reps+=("${m_reps[k]}")
      keep_sets+=("${m_sets[k]}")
    done
    m_name=(${keep_name[@]+"${keep_name[@]}"}); m_kind=(${keep_kind[@]+"${keep_kind[@]}"}); m_warm=(${keep_warm[@]+"${keep_warm[@]}"})
    m_ctx=(${keep_ctx[@]+"${keep_ctx[@]}"}); m_gen=(${keep_gen[@]+"${keep_gen[@]}"}); m_syn=(${keep_syn[@]+"${keep_syn[@]}"})
    m_prompt=(${keep_prompt[@]+"${keep_prompt[@]}"}); m_reps=(${keep_reps[@]+"${keep_reps[@]}"}); m_sets=(${keep_sets[@]+"${keep_sets[@]}"})
  fi
}

idx() { # NAME ARRAY...: position of NAME in the array
  local name=$1 i=0 x
  shift
  for x in "$@"; do
    [[ $x == "$name" ]] && { echo "$i"; return 0; }
    i=$((i + 1))
  done
  return 1
}

env_of() { local i; i=$(idx "$1" ${set_names[@]+"${set_names[@]}"}); echo "${set_envs[i]}"; }

if [[ ${DRY_RUN:-0} == 1 ]]; then
  total=0
  for step in $STEPS; do
    select_step "$step"
    printf 'step %s: model=%s sockets=%s generation=%s base: %s\n' "$step" "$model" "$sockets" "$generation" "${cfg[*]}"
    printf '  bind: %s\n' "${bind[*]-}"
    for ((i = 0; i < ${#set_names[@]}; i++)); do printf '  setting %s: %s\n' "${set_names[i]}" "${set_envs[i]}"; done
    for ((i = 0; i < ${#m_name[@]}; i++)); do
      n=$(wc -w <<< "${m_sets[i]}")
      printf '  measurement %s: %s warm=%s context=%s generation=%s synthetic=%s prompt=%s reps=%s settings=%s\n' "${m_name[i]}" \
        "${m_kind[i]}" "${m_warm[i]}" "${m_ctx[i]}" "${m_gen[i]}" "${m_syn[i]}" "${m_prompt[i]}" "${m_reps[i]}" "${m_sets[i]}"
      total=$((total + n * (${m_reps[i]} + 1)))
    done
  done
  printf 'runs (repetitions plus one profiled run per cell, before any plan gate): %s\n' "$total"
  exit 0
fi

# ---- job setup ----------------------------------------------------------------------------------------------------------------
mkdir "$OUT" "$OUT/plan"
exec > >(tee "$OUT/job.stdout") 2> >(tee "$OUT/job.stderr" >&2)
export LC_ALL=C
commit=$(cat "$ROOT/COMMIT" 2> /dev/null || git -C "$ROOT" rev-parse HEAD)
perf_bin=${PERF_BIN:-/usr/bin/perf}
# Intel PCM uses 32 bytes per HBM CAS on Sapphire Rapids and 64 bytes per DDR CAS.
reads='uncore_imc/event=0x05,umask=0xcf,name=ddr_cas_rd/,uncore_type_14/event=0x05,umask=0xcf,name=hbm_cas_rd/'
writes='uncore_imc/event=0x05,umask=0xf0,name=ddr_cas_wr/,uncore_type_14/event=0x05,umask=0xf0,name=hbm_cas_wr/'
events="$reads,$writes,unc_upi_rxl_flits.all_data,unc_upi_txl_flits.all_data,power/energy-pkg/,power/energy-ram/,cycles,ref-cycles"
interval=100
perf=(sudo -n "$perf_bin" stat -a --per-socket -I "$interval" -x, -e "$events")
{
  printf 'job=%s\ncommit=%s\nsteps=%s\nrepeats=%s\nrepeats_ab=%s\ninterval_ms=%s\n' "${SLURM_JOB_ID:-}" "$commit" "$STEPS" "$repeats" \
    "$repeats_ab" "$interval"
  date -Is; hostname; uname -a; gcc --version | head -1
  numactl -H
  grep -E '^(Cpus|Mems)_allowed_list' /proc/self/status
  lscpu
  printf 'transparent_hugepage: '; cat /sys/kernel/mm/transparent_hugepage/enabled || true
  printf 'numa_balancing: '; cat /proc/sys/kernel/numa_balancing || true
  printf 'cpu0 governor: '; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor || true
  "$perf_bin" --version || true
  scontrol show job "${SLURM_JOB_ID:-0}" || true
  printf 'models: 8b=%s 70b=%s\n' "${MODEL_8B:-}" "${MODEL_70B:-}"
  [[ -z ${MODEL_8B:-} ]] || stat -c '%n %s' "$MODEL_8B/int8-v3.bin" "$MODEL_8B/tokenizer-v2.bin"
  [[ -z ${MODEL_70B:-} ]] || stat -c '%n %s' "$MODEL_70B/int8-v3.bin" "$MODEL_70B/tokenizer-v2.bin"
  printf 'caller LLAMA_* (cleared before every cell): %s\n' "$(env | grep '^LLAMA_' | sort | tr '\n' ' ')"
  printf 'events=%s\n' "$events"
} > "$OUT/environment.txt" 2>&1
(cd "$ROOT" && find src include tests tools scripts Makefile -type f ! -path '*/__pycache__/*' -print0 | sort -z |
  xargs -0 sha256sum) > "$OUT/source.sha256"

# Every LLAMA_* variable of the caller is cleared, so a cell sees exactly the environment written down.
unset_args=()
while IFS= read -r name; do unset_args+=(-u "$name"); done < <(env | sed -n 's/^\(LLAMA_[A-Za-z0-9_]*\)=.*/\1/p')
clean() { env ${unset_args[@]+"${unset_args[@]}"} "$@"; }

# Builds: the normal one, the synthetic-K/V one where a step needs it, and the probe that counts tokens for the prompt files.
need_syn=0
for step in $STEPS; do
  select_step "$step"
  [[ $needs_syn == 0 ]] || need_syn=1
done
mkdir "$OUT/bin"
make -C "$ROOT" -j 56 CC=gcc BUILD_DIR="$OUT/bin" "$OUT/bin/runq" "$OUT/bin/probe-fp32" > "$OUT/build.log" 2>&1 ||
  { tail -30 "$OUT/build.log"; exit 1; }
if [[ $need_syn == 1 ]]; then
  mkdir "$OUT/bin-syn"
  make -C "$ROOT" -j 56 CC=gcc SYNTHETIC_KV=1 BUILD_DIR="$OUT/bin-syn" "$OUT/bin-syn/runq" > "$OUT/build-syn.log" 2>&1 ||
    { tail -30 "$OUT/build-syn.log"; exit 1; }
fi
sha256sum "$OUT/bin/runq" "$0" > "$OUT/executables.sha256"
[[ $need_syn == 0 ]] || sha256sum "$OUT/bin-syn/runq" >> "$OUT/executables.sha256"

# Prompt files of the TTFT cells: the longest prefix of the repeated paragraph whose count with the chat template (which the
# benchmark applies) does not exceed the target. One set per tokenizer; the achieved counts are in prompts-<model>/prompts.json.
make_prompts() { # MODEL TOKENIZER TARGETS
  "$PYTHON" - "$ROOT" "$OUT/bin/probe-fp32" "$2" "$OUT/prompts-$1" "$3" << 'PY'
import hashlib, json, subprocess, sys
from pathlib import Path
root, probe, tokenizer, out, targets = sys.argv[1:6]
sys.path.insert(0, root + '/tools')
import make_prompts as mp


class Chat(mp.Counter):
    def count(self, text):
        self.path.write_text(text, encoding='utf-8')
        done = subprocess.run([probe, 'chat', tokenizer, '128256', f'@{self.path}'], capture_output=True, text=True)
        if done.returncode:
            raise SystemExit(f'probe chat failed: {done.stderr}')
        return len(done.stdout.split())


out = Path(out)
out.mkdir(parents=True)
counter = Chat(probe, tokenizer, 128256)
manifest = {}
for target in (int(n) for n in targets.split(',')):
    text = mp.prefix_for(counter, mp.PARAGRAPH * (target * 8 // len(mp.PARAGRAPH) + 2), target)
    (out / f'prompt-{target}.txt').write_text(text, encoding='utf-8')
    manifest[f'prompt-{target}'] = dict(file=f'prompt-{target}.txt', tokens_with_chat_template=counter.count(text),
                                        characters=len(text), sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
(out / 'prompts.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest))
PY
}
# the lengths each model needs, from every step of that model
for model_name in 8b 70b; do
  targets=
  tok=
  for step in $STEPS; do
    select_step "$step"
    [[ $model == "$model_name" ]] || continue
    tok=$tokenizer
    for ((i = 0; i < ${#m_name[@]}; i++)); do
      [[ ${m_kind[i]} == ttft ]] || continue
      case ",$targets," in *",${m_prompt[i]},"*) ;; *) targets="${targets:+$targets,}${m_prompt[i]}" ;; esac
    done
  done
  [[ -z $targets ]] || make_prompts "$model_name" "$tok" "$targets" > "$OUT/prompts-$model_name.log"
done

# Counters: the job needs them (the CAS bytes are a deliverable).
sudo -n "$perf_bin" stat -a --per-socket -I 100 -x, -e "$events" -- sleep 1 > /dev/null 2> "$OUT/perf-check.txt" ||
  { echo "perf stat is not available through sudo" >&2; exit 1; }
"${perf[@]}" -- sleep 5 2> "$OUT/idle.perf.csv"

printf 'label\tstep\tsetting\tmeas\tkind\trep\twarm\tcontext\tsynthetic\tprompt_tokens\tgroup\tmodel\tsockets\tprofile\tstart_ns\tend_ns\n' > "$OUT/runs.tsv"
printf 'step\tsetting\tmeas\treason\n' > "$OUT/skipped.tsv"
printf 'label\tstatus\n' > "$OUT/failures.tsv"

# ---- plan gate ----------------------------------------------------------------------------------------------------------------
engine_bin() { [[ $1 == 1 ]] && echo "$OUT/bin-syn/runq" || echo "$OUT/bin/runq"; }

# plan_cell STEP SETTING MEAS: --plan-only of the cell; the resolved environment goes to plan/STEP-SETTING-MEAS.env
plan_cell() {
  local step=$1 setting=$2 meas=$3 i env_line plan kv status=0
  i=$(idx "$meas" "${m_name[@]}")
  env_line=$(env_of "$setting")
  if [[ $env_line == *@KVMATCH@* ]]; then # the K/V layers the serial cell of the same measurement was given
    plan="$OUT/plan/$step-serial-w80-$meas.json"
    [[ -s $plan ]] || { echo "kvmatch: no plan of serial-w80 for $meas" >&2; return 1; }
    kv=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["placement_plan"]["kv_hbm_layers"])' "$plan") || return 1
    env_line=${env_line//@KVMATCH@/$kv}
  fi
  printf '%s\n' "$env_line" > "$OUT/plan/$step-$setting-$meas.env"
  plan="$OUT/plan/$step-$setting-$meas"
  # shellcheck disable=SC2086
  clean "${cfg[@]}" $env_line ${bind[@]+"${bind[@]}"} "$(engine_bin "${m_syn[i]}")" "$checkpoint" --context-length "${m_ctx[i]}" --plan-only \
    --json "$plan.json" > "$plan.stdout" 2> "$plan.stderr" || status=$?
  [[ $status == 0 ]] || return 1
  "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["placement_plan"]["admitted"] is True else 1)' "$plan.json"
}

# gate_step: admit every (setting, measurement); the admitted settings of a measurement go to plan/STEP-MEAS.sets in order
gate_step() {
  local step=$1 i s admitted
  for ((i = 0; i < ${#m_name[@]}; i++)); do
    admitted=
    for s in ${m_sets[i]}; do
      if plan_cell "$step" "$s" "${m_name[i]}"; then
        admitted="$admitted $s"
      else
        printf '%s\t%s\t%s\tnot admitted by --plan-only (%s)\n' "$step" "$s" "${m_name[i]}" \
          "$(tail -c 300 "$OUT/plan/$step-$s-${m_name[i]}.stderr" 2> /dev/null | tr '\n\t' '  ')" >> "$OUT/skipped.tsv"
        printf '%s plan gate: %s %s %s not admitted\n' "$(date -Is)" "$step" "$s" "${m_name[i]}"
      fi
    done
    printf '%s\n' "$admitted" > "$OUT/plan/$step-${m_name[i]}.sets"
  done
}

# ---- one run ------------------------------------------------------------------------------------------------------------------
# run_cell LABEL STEP SETTING MEAS REP PROFILE: one benchmark process with the uncore counters, then its checks
run_cell() {
  local label=$1 step=$2 setting=$3 meas=$4 rep=$5 prof=$6
  local i prefix=$OUT/$label env_line status=0 pid perfpid group kern=default prompt_args syn_env=() extra=() prompt_tokens=0 start end
  i=$(idx "$meas" "${m_name[@]}")
  env_line=$(cat "$OUT/plan/$step-$setting-$meas.env")
  [[ $env_line != *LLAMA_KERNELS=avx512* ]] || kern=avx512
  group="$model-${sockets}S-$meas-$kern"
  if [[ ${m_kind[i]} == decode ]]; then
    prompt_args=(--prompt "$prime_prompt")
  else
    prompt_args=(--prompt-file "$OUT/prompts-$model/prompt-${m_prompt[i]}.txt")
    prompt_tokens=${m_prompt[i]}
  fi
  [[ ${m_syn[i]} == 0 ]] || syn_env=(LLAMA_SYNTHETIC_KV=1 "LLAMA_SYNTHETIC_KV_PRIME=$syn_prime")
  [[ $prof == 0 ]] || extra=(LLAMA_PROFILE=1)
  printf '%s %s\n' "$(date -Is)" "$label"
  numactl -H | grep -E '^node [0-3] free' > "$prefix.numa-free" || true
  date +%s%N > "$prefix.launch"
  start=$(date +%s%N)
  # shellcheck disable=SC2086
  clean "${cfg[@]}" $env_line ${syn_env[@]+"${syn_env[@]}"} ${extra[@]+"${extra[@]}"} LLAMA_TOKENS_OUT="$prefix.tokens.json" \
    ${bind[@]+"${bind[@]}"} "$(engine_bin "${m_syn[i]}")" "$checkpoint" -z "$tokenizer" --benchmark --chat-template --no-text -t 0 -n 0 \
    --context-length "${m_ctx[i]}" --warm-context "${m_warm[i]}" --generation-length "${m_gen[i]}" "${prompt_args[@]}" \
    --json "$prefix.json" > "$prefix.stdout" 2> "$prefix.stderr" &
  pid=$!
  date +%s%N > "$prefix.perf-start"
  "${perf[@]}" -- tail -s 0.05 --pid="$pid" -f /dev/null > /dev/null 2> "$prefix.perf.csv" &
  perfpid=$!
  wait "$pid" || status=$?
  end=$(date +%s%N)
  printf '%s\n' "$end" > "$prefix.end"
  wait "$perfpid" || true
  date +%s%N > "$prefix.perf-end"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$step" "$setting" "$meas" "${m_kind[i]}" "$rep" \
    "${m_warm[i]}" "${m_ctx[i]}" "${m_syn[i]}" "$prompt_tokens" "$group" "$model" "$sockets" "$prof" "$start" "$end" >> "$OUT/runs.tsv"
  if [[ $status != 0 ]]; then
    tail -20 "$prefix.stderr" >&2
    return "$status"
  fi
  "$PYTHON" "$ROOT/tools/check_placement.py" "$prefix.json" --planned --allow-sampled > "$prefix.placement" ||
    { cat "$prefix.placement" >&2; return 1; }
  "$PYTHON" - "$prefix.json" "${m_gen[i]}" "${m_syn[i]}" "${m_kind[i]}" "$prefix.tokens.json" << 'PY'
import json, sys
report = json.load(open(sys.argv[1]))
expected = int(sys.argv[2]) - 1
if bool(report["parameters"].get("synthetic_kv")) != bool(int(sys.argv[3])):
    raise SystemExit(f"{sys.argv[1]}: unexpected KV population method")
if report["stop_reason"] not in ("generation_limit", "step_limit", "end_of_text", "eos"):
    raise SystemExit(f"{sys.argv[1]}: stop {report['stop_reason']}")
# the model may end its answer before the limit: a decode cell needs at least min(expected, 64) timed decode steps
if sys.argv[4] == "decode" and report["tokens"]["decode_steps"] < min(expected, 64):
    raise SystemExit(f"{sys.argv[1]}: {report['tokens']['decode_steps']} decode steps (wanted {expected})")
if sys.argv[4] == "ttft" and report["tokens"]["processed_prompt"] < 1000:
    raise SystemExit(f"{sys.argv[1]}: only {report['tokens']['processed_prompt']} prompt tokens")
PY
}

consecutive=0
failed=0
attempt() { # the arguments of run_cell; a failed run is recorded and the job goes on, unless three fail in a row
  if run_cell "$@"; then
    consecutive=0
  else
    printf '%s\tfailed\n' "$1" >> "$OUT/failures.tsv"
    printf '%s FAILED %s\n' "$(date -Is)" "$1" >&2
    consecutive=$((consecutive + 1))
    failed=$((failed + 1))
    [[ $consecutive -lt 3 ]] || { echo "three runs failed in a row: stopping" >&2; exit 1; }
  fi
}

run_step() {
  local step=$1 rep max=0 i s list n k
  for ((i = 0; i < ${#m_name[@]}; i++)); do [[ ${m_reps[i]} -le $max ]] || max=${m_reps[i]}; done
  for ((rep = 1; rep <= max; rep++)); do
    for ((i = 0; i < ${#m_name[@]}; i++)); do
      [[ $rep -le ${m_reps[i]} ]] || continue
      read -r -a list < "$OUT/plan/$step-${m_name[i]}.sets" || list=()
      n=${#list[@]}
      for ((k = 0; k < n; k++)); do # repetition r starts at cell r mod n: every cell meets every neighbour
        s=${list[(rep - 1 + k) % n]}
        attempt "r$rep-$step-$s-${m_name[i]}" "$step" "$s" "${m_name[i]}" "$rep" 0
      done
    done
  done
  for ((i = 0; i < ${#m_name[@]}; i++)); do # one profiled run per cell
    read -r -a list < "$OUT/plan/$step-${m_name[i]}.sets" || list=()
    for s in ${list[@]+"${list[@]}"}; do
      attempt "prof-$step-$s-${m_name[i]}" "$step" "$s" "${m_name[i]}" 0 1
    done
  done
}

# Token traces: every run of one group (model, sockets, measurement, kernels) must have the same trace
check_traces() {
  "$PYTHON" - "$OUT" << 'PY'
import csv, json, sys
from pathlib import Path
out = Path(sys.argv[1])
groups = {}
for row in csv.DictReader(open(out / 'runs.tsv'), delimiter='\t'):
    path = out / f"{row['label']}.tokens.json"
    if path.exists():
        groups.setdefault(row['group'], {})[row['label']] = json.loads(path.read_text())
lines, bad = [], 0
for group, traces in sorted(groups.items()):
    reference = next(iter(traces.values()))
    differ = [label for label, trace in traces.items() if trace != reference]
    bad += bool(differ)
    lines.append(f'{group}: {len(traces)} traces, {len(differ)} differ{": " + " ".join(differ) if differ else ""}')
(out / 'tokens-identical.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
sys.exit(1 if bad else 0)
PY
}

for step in $STEPS; do
  select_step "$step"
  printf '%s step %s: %s settings, %s measurements\n' "$(date -Is)" "$step" "${#set_names[@]}" "${#m_name[@]}"
  gate_step "$step"
done
if [[ ${PLAN_ONLY:-0} == 1 ]]; then
  cat "$OUT/skipped.tsv"
  echo "plan only: finished"
  exit 0
fi
for step in $STEPS; do
  select_step "$step"
  run_step "$step"
done
status=0
check_traces || status=1
[[ $failed == 0 ]] || status=1
printf 'finished %s, %s failed runs\n' "$(date -Is)" "$failed"
exit "$status"
