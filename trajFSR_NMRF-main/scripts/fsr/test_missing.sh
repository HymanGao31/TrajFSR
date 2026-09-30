#!/usr/bin/env bash
# Missing-frame (LOCF) eval for NMRF groups A/B/C. Does not change noise train/test.
# Do NOT use env GROUP — AutoDL sets GROUP=0. Use FSR_GROUP instead.
#   bash scripts/fsr/test_missing.sh
#   DATASET=eth FSR_GROUP=C bash scripts/fsr/test_missing.sh
#   DATASET=sdd FSR_GROUP=C bash scripts/fsr/test_missing.sh
#   DATASET=jrdb FSR_GROUP=C bash scripts/fsr/test_missing.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

DATASET="${DATASET:-}"
FSR_GROUP="${FSR_GROUP:-}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
RATIOS="${RATIOS:-0.1,0.2,0.3}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/nmrfFSR}"
CKPT="${CKPT:-}"
TRAIN_NOISE="${TRAIN_NOISE:-0.1}"

if [[ -z "${DATASET}" ]]; then
  DATASETS=(eth hotel univ zara1 zara2 nba sdd)
else
  DATASETS=("${DATASET}")
fi
case "${FSR_GROUP}" in
  A|a) GROUPS=(A) ;;
  B|b) GROUPS=(B) ;;
  C|c) GROUPS=(C) ;;
  *)   GROUPS=(A B C) ;;
esac

for ds in "${DATASETS[@]}"; do
  for g in "${GROUPS[@]}"; do
    echo "===== MISSING-LOCF  dataset=${ds}  group=${g}  ratios=${RATIOS} ====="
    extra=()
    [[ -n "${CKPT}" ]] && extra+=(--ckpt "${CKPT}")
    python scripts/eval_missing.py \
      --dataset "${ds}" \
      --group "${g}" \
      --missing_ratios "${RATIOS}" \
      --seed "${SEED}" \
      --gpu "${GPU}" \
      --results_root "${RESULTS_ROOT}" \
      --train_noise "${TRAIN_NOISE}" \
      "${extra[@]}"
  done
done
