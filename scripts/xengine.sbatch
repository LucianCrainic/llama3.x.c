#!/usr/bin/env bash
#SBATCH --job-name=xengine
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:00:00
# The cross-engine campaign job (llama3.x.c, llama.cpp, vLLM CPU) on one CRESCO8 Xeon Max node; the plan and the rules are
# in docs/plans/xengine-comparison.md. Separate from scripts/cresco8.sbatch, which it leaves untouched; the steps live in
# scripts/lib/xengine-common.sh. Submit with a verified --partition/--nodelist and separate -o/-e paths.
#
# Required environment: ROOT (staged copy of this repository), OUT (new campaign directory), NODE_ROLE (A = Flat node,
# everything on HBM; B = Flat node, everything on DDR plus the c / 1S / HBM / D anchor; cache = Cache-mode node, timing
# first, no privileged counters).
# Optional: ENGINES (subset of "c llamacpp vllm", default c), SOCKETS ("1 2"), MEMS (hbm | ddr | cache; default by role),
#   WORKLOADS ("P D L": P = 2048-token prompt with G = 1, D = 86-token prompt with G = 128, L = 8192-token prompt with
#   G = 128), REPEATS (3: repeat 1 = memory and power counters, 2 = compute, AMX and power counters, 3 = no perf),
#   DRY_RUN=1 (print the whole matrix of commands and a time estimate, run nothing), PYTHON (python3), ANCHOR, UPERF.
# Engine inputs: ARTIFACTS (directory with int8-v3.bin and tokenizer-v2.bin), PROMPTS_DIR (short-86.txt, prompt-2048.txt,
#   prompt-8192.txt), C_ENGINE; GGUF, LLAMACPP_BIN and VLLM_* when those engines are filled in.
# Output: OUT/{environment.txt, manifest.json, runs.csv, idle-baseline.perf.csv, phase-results.txt, runs/<label>/...}. The
# exit status is non-zero if any cell failed; a failed cell is recorded and the job goes on.
set -euo pipefail
# Slurm runs a copy of this script, so the library is found through ROOT, not through the script's own path.
: "${ROOT:?staged repository copy required}"
# shellcheck source=lib/xengine-common.sh
source "$ROOT/scripts/lib/xengine-common.sh"
xe_require_env
xe_job_begin
xe_record_environment
xe_check_node
xe_perf_available
xe_idle_baseline
for engine in $ENGINES; do
  xe_run_engine "$engine"
done
xe_print_estimate
exit "$OVERALL"
