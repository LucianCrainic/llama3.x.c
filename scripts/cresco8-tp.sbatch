#!/usr/bin/env bash
#SBATCH --job-name=llama-tp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
# Flat Mode campaign skeleton for tensor parallelism (track B) on a dual Xeon Max 9480 node.
# Submit with a verified --partition/--nodelist and separate -o/-e paths; every submission needs the
# owner's go-ahead for that job. Required: ROOT (tree under test), OUT (new directory), PYTHON, MODELS
# ("NAME:ARTIFACT_DIR:HF_SNAPSHOT ..."; the artifact directory holds fp32-v3.bin, int8-v3.bin, tokenizer-v2.bin).
# STAGE=check builds, tests and validates only; STAGE=bench (default) adds the PHASE run matrix.
# PHASE=none runs only the common check. A phase adds its matrix to the case below when it is implemented.
set -euo pipefail
# Slurm runs a copy of this script, so the library is found through ROOT, not through the script's own path.
: "${ROOT:?tree under test required}"
# shellcheck source=lib/cresco8-common.sh
source "$ROOT/scripts/lib/cresco8-common.sh"
require_env
job_begin
check_stage
validate_models
[[ $STAGE == bench ]] || exit 0

case "$PHASE" in
  none) echo "PHASE=none: no benchmark matrix selected" ;;
  *) die "phase $PHASE has no run matrix in $(basename "${BASH_SOURCE[0]}") yet" ;;
esac
