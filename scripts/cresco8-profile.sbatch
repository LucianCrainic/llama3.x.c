#!/bin/bash
#SBATCH --job-name=llama-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=01:30:00
# Per-phase decode profile (LLAMA_PROFILE=1) on one flat-mode socket with
# weights on local HBM. Requires ROOT (source tree), OUT and MODELS, a
# space-separated list of NAME:ARTIFACTS entries. Each configuration also runs
# once unprofiled to bound the profiler's own cost, and an empty OpenMP region
# loop measures the fork/join floor.
set -euo pipefail
: "${ROOT:?}" "${OUT:?}" "${MODELS:?NAME:ARTIFACTS entries required}"
REPEATS=${REPEATS:-2}
mkdir -p "$OUT/unprofiled"
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
{ date -u +%FT%TZ; hostname; cat "$ROOT/COMMIT" 2>/dev/null || git -C "$ROOT" rev-parse HEAD; } > "$OUT/environment.txt"
(cd "$ROOT" && numactl --cpunodebind=0 --membind=0 make CC=gcc OPENMP=1 build/release/run build/release/runq) \
  > "$OUT/build.log" 2>&1

cat > "$OUT/fork-join.c" <<'EOF'
#include <omp.h>
#include <stdio.h>
int main(void) {
  for (int pass = 0; pass < 2; pass++) {
    double t0 = omp_get_wtime();
    for (int i = 0; i < 100000; i++) {
#pragma omp parallel
      {
      }
    }
    if (pass)
      printf("threads=%d parallel-region=%.2f us\n", omp_get_max_threads(), (omp_get_wtime() - t0) * 10.0);
  }
  return 0;
}
EOF
gcc -O2 -fopenmp "$OUT/fork-join.c" -o "$OUT/fork-join"
for threads in 14 28 56; do
  OMP_NUM_THREADS=$threads numactl --cpunodebind=0 --membind=0 "$OUT/fork-join"
done > "$OUT/fork-join.txt"

sentence='Explain how a CPU executes instructions and uses memory. '
printf '%s%s%s%s%s' "$sentence" "$sentence" "$sentence" "$sentence" "$sentence" > "$OUT/short.txt"
shm=$(mktemp -d "/dev/shm/llama-profile-$SLURM_JOB_ID-XXXXXX")
trap 'rm -rf "$shm"' EXIT
run() { # label engine checkpoint tokenizer threads repeats [env...]
  local label=$1 engine=$2 checkpoint=$3 tokenizer=$4 threads=$5 repeats=$6
  shift 6
  for ((r = 1; r <= repeats; r++)); do
    env "$@" OMP_NUM_THREADS="$threads" numactl --cpunodebind=0 --membind=2 "$ROOT/build/release/$engine" \
      "$checkpoint" -z "$tokenizer" --context-length 2048 --benchmark --chat-template --prompt-file "$OUT/short.txt" \
      -t 0 --seed 42 --generation-length 128 --no-text --log-level warn --json "$OUT/$label-$r.json" \
      > /dev/null 2> "$OUT/$label-$r.stderr"
  done
}
for entry in $MODELS; do
  IFS=: read -r name artifacts <<< "$entry"
  for precision in fp32 int8; do
    engine=run; [[ $precision == fp32 ]] || engine=runq
    checkpoint="$shm/$name-$precision.bin"
    numactl --membind=2 cp "$artifacts/$precision-v3.bin" "$checkpoint"
    for threads in 56 14; do
      repeats=$REPEATS; [[ $threads == 56 ]] || repeats=1
      run "$name-$precision-t$threads" "$engine" "$checkpoint" "$artifacts/tokenizer-v2.bin" "$threads" "$repeats" \
        LLAMA_PROFILE=1
    done
    run "unprofiled/$name-$precision-t56" "$engine" "$checkpoint" "$artifacts/tokenizer-v2.bin" 56 1
    rm -f "$checkpoint"
  done
done
python3 "$ROOT/tools/summarize_profile.py" "$OUT" > "$OUT/summary.md"
