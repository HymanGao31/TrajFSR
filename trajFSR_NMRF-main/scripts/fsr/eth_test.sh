#!/usr/bin/env bash
# ETH/UCY test.
#   GROUP=B SUBSET=eth bash scripts/fsr/eth_test.sh
#   GROUP=C SUBSET=eth bash scripts/fsr/eth_test.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

SUBSET="${SUBSET:-hotel}"       # eth | hotel | univ | zara1 | zara2
GROUP="${GROUP:-C}"             # A | B | C
GPU="${GPU:-0}"
TRAIN_NOISE="${TRAIN_NOISE:-0.1}"
FREEZE="${FREEZE:-0}"
FSR_SCALE="${FSR_SCALE:-0.2}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/nmrfFSR}"
CKPT="${CKPT:-}"
EXP="${EXP:-}"

official_epoch () {
  case "$1" in
    eth) echo 47 ;;
    hotel) echo 44 ;;
    univ) echo 15 ;;
    zara1) echo 47 ;;
    zara2) echo 55 ;;
    *) echo 0 ;;
  esac
}

c_tag () {
  local s="FSR"
  [[ "${FREEZE}" == "1" ]] && s="${s}_frz"
  [[ "${FSR_SCALE}" != "1.0" ]] && s="${s}_fsc${FSR_SCALE}"
  echo "${s}_ntr${TRAIN_NOISE}"
}

case "${GROUP}" in
  A) CKPT="${CKPT:-results/${SUBSET}/default/model_$(printf '%04d' "$(official_epoch "${SUBSET}")").p}"
     EXP="${EXP:-}" ;;
  B) CKPT="${CKPT:-${RESULTS_ROOT}/${SUBSET}_nmrf_eth_noiseB_ntr${TRAIN_NOISE}/${SUBSET}_ckpt_best.pth}"
     EXP="${EXP:-nmrf_eth_noiseB}" ;;
  C)
     EXP="${EXP:-nmrf_eth_fsrC_onB}"
     if [[ -z "${CKPT}" ]]; then
       # Existing C dirs may still have _id0_ from older runs.
       for tag in "$(c_tag)" "FSR_id0_fsc${FSR_SCALE}_ntr${TRAIN_NOISE}" "FSR_fsc${FSR_SCALE}_ntr${TRAIN_NOISE}"; do
         cand="${RESULTS_ROOT}/${SUBSET}_${EXP}_${tag}/${SUBSET}_ckpt_best.pth"
         if [[ -f "${cand}" ]]; then CKPT="${cand}"; break; fi
       done
     fi
     ;;
  *) echo "GROUP must be A, B, or C"; exit 1 ;;
esac
[[ -n "${CKPT}" && -f "${CKPT}" ]] || {
  echo "missing ckpt (tried C-on-B dirs under ${RESULTS_ROOT}/${SUBSET}_${EXP:-nmrf_eth_fsrC_onB}_*)"
  exit 1
}
echo "[INFO] TEST ckpt: ${CKPT}"

base=(python main.py --dataset "${SUBSET}" --gpu "${GPU}" --ckpt_path "${CKPT}")
if [[ "${GROUP}" == "C" ]]; then
  base+=(--fsr --exp "${EXP}" --results_root "${RESULTS_ROOT}" --train_noise_sigma "${TRAIN_NOISE}")
elif [[ "${GROUP}" == "B" ]]; then
  base+=(--exp "${EXP}" --results_root "${RESULTS_ROOT}" --train_noise_sigma "${TRAIN_NOISE}")
fi

run_grid () {
  local SM=$1 SIG
  for SIG in 0 0.1 0.2 0.3 0.4 0.5; do
    echo "===== TEST ${GROUP} | ${SUBSET} | noise=${SIG} | smooth=${SM} ====="
    "${base[@]}" --eval_noise_sigma "${SIG}" --smooth_past_deg "${SM}"
  done
}

run_grid -1
run_grid 2
