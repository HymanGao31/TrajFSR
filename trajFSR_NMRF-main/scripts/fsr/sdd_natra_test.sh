#!/usr/bin/env bash
# NATRA-style cross-noise eval on SDD (eval only, no retraining).
# Meter units, same numbers as ETH NATRA:
#   poisson λ=0.4 | mixed σ=0.2+λ=0.2 | rand_sigma {0.2,0.4}
# Variants: group x {no-smooth, smooth=2}. One group per run.
#
#   FSR_GROUP=A bash scripts/fsr/sdd_natra_test.sh
#   FSR_GROUP=B bash scripts/fsr/sdd_natra_test.sh
#   FSR_GROUP=C bash scripts/fsr/sdd_natra_test.sh
#
# C defaults to the working C-on-B (FREEZE=1 MASK_BAL=2). Do NOT use AutoDL GROUP=0.
# Existing sdd_train.sh / sdd_test.sh (Gaussian grid) are unchanged.
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -n "${FSR_GROUP:-}" ]]; then
  PAPER_GROUP="${FSR_GROUP}"
else
  case "${GROUP:-}" in
    A|a|B|b|C|c) PAPER_GROUP="${GROUP}" ;;
    *)
      echo "[ERR] AutoDL GROUP='${GROUP:-}' is not A/B/C. Use FSR_GROUP=A|B|C"
      exit 1
      ;;
  esac
fi
PAPER_GROUP="$(echo "${PAPER_GROUP}" | tr '[:lower:]' '[:upper:]')"

GPU="${GPU:-0}"
TRAIN_NOISE="${TRAIN_NOISE:-0.1}"
FREEZE="${FREEZE:-1}"
FSR_SCALE="${FSR_SCALE:-0.2}"
MASK_BAL="${MASK_BAL:-2}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/nmrfFSR}"
CKPT="${CKPT:-}"
EXP="${EXP:-}"
SMOOTH_DEG="${SMOOTH_DEG:-2}"
PROTOCOLS="${PROTOCOLS:-poisson,mixed,rand_sigma}"

c_tag () {
  local s="FSR"
  if [[ "${FREEZE}" == "enc" ]]; then
    s="${s}_frzE"
  elif [[ "${FREEZE}" == "1" ]]; then
    s="${s}_frz"
  fi
  [[ "${MASK_BAL}" != "0" && "${MASK_BAL}" != "0.0" ]] && s="${s}_mb${MASK_BAL}"
  [[ "${FSR_SCALE}" != "1.0" ]] && s="${s}_fsc${FSR_SCALE}"
  echo "${s}_ntr${TRAIN_NOISE}"
}

case "${PAPER_GROUP}" in
  A) CKPT="${CKPT:-results/sdd/default/model_0080.p}"
     EXP="${EXP:-}" ;;
  B) CKPT="${CKPT:-${RESULTS_ROOT}/sdd_nmrf_sdd_noiseB_ntr${TRAIN_NOISE}/sdd_ckpt_best.pth}"
     EXP="${EXP:-nmrf_sdd_noiseB}" ;;
  C)
     EXP="${EXP:-nmrf_sdd_fsrC_onB}"
     if [[ -z "${CKPT}" ]]; then
       for tag in "$(c_tag)" "FSR_frz_mb2_fsc${FSR_SCALE}_ntr${TRAIN_NOISE}"; do
         cand="${RESULTS_ROOT}/sdd_${EXP}_${tag}/sdd_ckpt_best.pth"
         if [[ -f "${cand}" ]]; then CKPT="${cand}"; break; fi
       done
     fi
     ;;
  *) echo "FSR_GROUP must be A, B, or C"; exit 1 ;;
esac
[[ -n "${CKPT}" && -f "${CKPT}" ]] || {
  echo "missing ckpt: ${CKPT:-<empty>}"
  exit 1
}
echo "[INFO] NATRA SDD ${PAPER_GROUP} ckpt: ${CKPT}"

base=(python main.py --dataset sdd --gpu "${GPU}" --ckpt_path "${CKPT}" --eval_noise_sigma 0)
if [[ "${PAPER_GROUP}" == "C" ]]; then
  base+=(--fsr --exp "${EXP}" --results_root "${RESULTS_ROOT}" --train_noise_sigma "${TRAIN_NOISE}")
elif [[ "${PAPER_GROUP}" == "B" ]]; then
  base+=(--exp "${EXP}" --results_root "${RESULTS_ROOT}" --train_noise_sigma "${TRAIN_NOISE}")
fi

IFS=',' read -r -a PROTO_ARR <<< "${PROTOCOLS}"
for proto in "${PROTO_ARR[@]}"; do
  proto="$(echo "${proto}" | tr -d '[:space:]')"
  [[ -n "${proto}" ]] || continue
  for sm in -1 "${SMOOTH_DEG}"; do
    echo "===== NATRA SDD ${PAPER_GROUP} | proto=${proto} | smooth=${sm} ====="
    "${base[@]}" --noise_protocol "${proto}" --smooth_past_deg "${sm}"
  done
done
