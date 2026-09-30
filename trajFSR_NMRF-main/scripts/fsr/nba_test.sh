#!/usr/bin/env bash
# NBA test. C
#   GROUP=A bash scripts/fsr/nba_test.sh
#   GROUP=B bash scripts/fsr/nba_test.sh
#   GROUP=C bash scripts/fsr/nba_test.sh
# Always runs no-smooth then smooth=2 on the noise grid.
set -euo pipefail
cd "$(dirname "$0")/../.."

GROUP="${GROUP:-C}"             # A | B | C
GPU="${GPU:-0}"
TRAIN_NOISE="${TRAIN_NOISE:-0.1}"
FREEZE="${FREEZE:-0}"
FSR_SCALE="${FSR_SCALE:-1.0}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/nmrfFSR}"
CKPT="${CKPT:-}"
EXP="${EXP:-}"

# 必须与 fsr_helpers.build_save_dir 顺序一致：FSR, sem, band, frz/frzE, id, mb, fsc, ntr
c_tag () {
  local s="FSR_sem_band"
  [[ "${FREEZE}" == "1" ]] && s="${s}_frz"
  [[ "${FREEZE}" == "enc" ]] && s="${s}_frzE"
  [[ "${FSR_SCALE}" != "1.0" ]] && s="${s}_fsc${FSR_SCALE}"
  echo "${s}_ntr${TRAIN_NOISE}"
}

case "${GROUP}" in
  A) CKPT="${CKPT:-results/nba/default/model_0014.p}"
     EXP="${EXP:-}" ;;
  B) CKPT="${CKPT:-${RESULTS_ROOT}/nba_nmrf_nba_noiseB_ntr${TRAIN_NOISE}/nba_ckpt_best.pth}"
     EXP="${EXP:-nmrf_nba_noiseB}" ;;
  C)
     EXP="${EXP:-nmrf_nba_fsrC_sem}"
     if [[ -z "${CKPT}" ]]; then
       cand="${RESULTS_ROOT}/nba_${EXP}_$(c_tag)/nba_ckpt_best.pth"
       if [[ -f "${cand}" ]]; then
         CKPT="${cand}"
       else
         CKPT="$(ls -t ${RESULTS_ROOT}/nba_${EXP}_FSR_*/nba_ckpt_best.pth 2>/dev/null | head -n 1 || true)"
       fi
     fi
     ;;
  *) echo "GROUP must be A, B, or C"; exit 1 ;;
esac
[[ -n "${CKPT}" && -f "${CKPT}" ]] || {
  echo "missing ckpt (tried under ${RESULTS_ROOT}/nba_${EXP:-nmrf_nba_fsrC_sem}_*)"
  exit 1
}
echo "[INFO] TEST NBA ${GROUP} ckpt: ${CKPT}"

base=(python main.py --dataset nba --gpu "${GPU}" --ckpt_path "${CKPT}")
if [[ "${GROUP}" == "C" ]]; then
  base+=(--fsr --fsr_sem --exp "${EXP}" --results_root "${RESULTS_ROOT}" --train_noise_sigma "${TRAIN_NOISE}")
elif [[ "${GROUP}" == "B" ]]; then
  base+=(--exp "${EXP}" --results_root "${RESULTS_ROOT}" --train_noise_sigma "${TRAIN_NOISE}")
fi

run_grid () {
  local SM=$1 SIG
  for SIG in 0 0.1 0.2 0.3 0.4 0.5; do
    echo "===== TEST NBA ${GROUP} | noise=${SIG} | smooth=${SM} ====="
    "${base[@]}" --eval_noise_sigma "${SIG}" --smooth_past_deg "${SM}"
  done
}

run_grid -1
run_grid 2
