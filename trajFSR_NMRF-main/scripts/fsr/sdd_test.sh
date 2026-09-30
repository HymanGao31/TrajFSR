#!/usr/bin/env bash
# SDD test for groups A / B / C. Noise grid, then C+smooth=2 like ETH.
#   FSR_GROUP=A bash scripts/fsr/sdd_test.sh
#   FSR_GROUP=B bash scripts/fsr/sdd_test.sh
#   FSR_GROUP=C bash scripts/fsr/sdd_test.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

if [[ -n "${FSR_GROUP:-}" ]]; then
  GROUP="${FSR_GROUP}"
else
  case "${GROUP:-}" in
    A|a|B|b|C|c) GROUP="${GROUP}" ;;
    "") GROUP="C" ;;
    *)
      echo "[ERR] AutoDL GROUP='${GROUP}' is not A/B/C. Use FSR_GROUP=A|B|C"
      exit 1
      ;;
  esac
fi
GROUP="$(echo "${GROUP}" | tr '[:lower:]' '[:upper:]')"

GPU="${GPU:-0}"
TRAIN_NOISE="${TRAIN_NOISE:-0.1}"
FREEZE="${FREEZE:-0}"
FSR_SCALE="${FSR_SCALE:-0.2}"
MASK_BAL="${MASK_BAL:-0}"
SEM="${SEM:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/nmrfFSR}"
CKPT="${CKPT:-}"
EXP="${EXP:-}"

c_tag () {
  # Must match fsr_helpers.build_save_dir: FSR, sem, band, frz/frzE, mb, fsc, ntr
  local s="FSR"
  [[ "${SEM}" == "1" ]] && s="${s}_sem_band"
  if [[ "${FREEZE}" == "enc" ]]; then
    s="${s}_frzE"
  elif [[ "${FREEZE}" == "1" ]]; then
    s="${s}_frz"
  fi
  [[ "${MASK_BAL}" != "0" && "${SEM}" != "1" ]] && s="${s}_mb${MASK_BAL}"
  [[ "${FSR_SCALE}" != "1.0" ]] && s="${s}_fsc${FSR_SCALE}"
  echo "${s}_ntr${TRAIN_NOISE}"
}

case "${GROUP}" in
  A) CKPT="${CKPT:-results/sdd/default/model_0080.p}"
     EXP="${EXP:-}" ;;
  B) CKPT="${CKPT:-${RESULTS_ROOT}/sdd_nmrf_sdd_noiseB_ntr${TRAIN_NOISE}/sdd_ckpt_best.pth}"
     EXP="${EXP:-nmrf_sdd_noiseB}" ;;
  C)
     if [[ "${SEM}" == "1" ]]; then
       EXP="${EXP:-nmrf_sdd_fsrC_sem}"
     else
       EXP="${EXP:-nmrf_sdd_fsrC_onB}"
     fi
     if [[ -z "${CKPT}" ]]; then
       for tag in "$(c_tag)" "FSR_frz_mb2_fsc${FSR_SCALE}_ntr${TRAIN_NOISE}" "FSR_id0_fsc${FSR_SCALE}_ntr${TRAIN_NOISE}" "FSR_fsc${FSR_SCALE}_ntr${TRAIN_NOISE}"; do
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
echo "[INFO] TEST SDD ${GROUP} ckpt: ${CKPT}"

base=(python main.py --dataset sdd --gpu "${GPU}" --ckpt_path "${CKPT}")
if [[ "${GROUP}" == "C" ]]; then
  base+=(--fsr --exp "${EXP}" --results_root "${RESULTS_ROOT}" --train_noise_sigma "${TRAIN_NOISE}")
  [[ "${SEM}" == "1" ]] && base+=(--fsr_sem)
elif [[ "${GROUP}" == "B" ]]; then
  base+=(--exp "${EXP}" --results_root "${RESULTS_ROOT}" --train_noise_sigma "${TRAIN_NOISE}")
fi

run_grid () {
  local SM=$1 SIG
  for SIG in 0 0.1 0.2 0.3 0.4 0.5; do
    echo "===== TEST SDD ${GROUP} | noise=${SIG} | smooth=${SM} ====="
    "${base[@]}" --eval_noise_sigma "${SIG}" --smooth_past_deg "${SM}"
  done
}

run_grid -1
run_grid 2
