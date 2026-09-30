#!/usr/bin/env bash
# ETH/UCY train.
#   GROUP=B SUBSET=eth bash scripts/fsr/eth_train.sh
#   GROUP=C SUBSET=eth bash scripts/fsr/eth_train.sh   # C-on-B
set -euo pipefail
cd "$(dirname "$0")/../.."

SUBSET="${SUBSET:-hotel}"       # eth | hotel | univ | zara1 | zara2
GROUP="${GROUP:-C}"             # B | C  (A 没有训练)
GPU="${GPU:-0}"
TRAIN_NOISE="${TRAIN_NOISE:-0.1}"
EPOCHS="${EPOCHS:-100}"
FREEZE="${FREEZE:-0}"
FSR_SCALE="${FSR_SCALE:-0.2}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/nmrfFSR}"
INIT_CKPT="${INIT_CKPT:-}"
TEACHER_CKPT="${TEACHER_CKPT:-}"
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

cmd=(python main.py --dataset "${SUBSET}" --train --gpu "${GPU}"
     --load_full
     --train_noise_sigma "${TRAIN_NOISE}" --eval_noise_sigma 0.2
     --results_root "${RESULTS_ROOT}")
[[ "${EPOCHS}" != "-1" ]] && cmd+=(--num_epochs "${EPOCHS}")

case "${GROUP}" in
  B)
    echo "===== TRAIN B | ${SUBSET} | ntr=${TRAIN_NOISE} ====="
    cmd+=(--exp "${EXP:-nmrf_eth_noiseB}")
    [[ -n "${INIT_CKPT}" ]] && cmd+=(--ckpt_path "${INIT_CKPT}")
    ;;
  C)
    # 接到已训好的 B 上；teacher 仍是干净官方权重
    if [[ -z "${INIT_CKPT}" ]]; then
      INIT_CKPT="${RESULTS_ROOT}/${SUBSET}_nmrf_eth_noiseB_ntr${TRAIN_NOISE}/${SUBSET}_ckpt_best.pth"
    fi
    if [[ -z "${TEACHER_CKPT}" ]]; then
      TEACHER_CKPT="results/${SUBSET}/default/model_$(printf '%04d' "$(official_epoch "${SUBSET}")").p"
    fi
    [[ -f "${INIT_CKPT}" ]] || { echo "missing Group B ckpt: ${INIT_CKPT}"; echo "train B first: GROUP=B SUBSET=${SUBSET} bash scripts/fsr/eth_train.sh"; exit 1; }
    [[ -f "${TEACHER_CKPT}" ]] || { echo "missing teacher: ${TEACHER_CKPT}"; exit 1; }
    echo "===== TRAIN C-on-B | ${SUBSET} | ntr=${TRAIN_NOISE} | freeze=${FREEZE} | fsc=${FSR_SCALE} ====="
    echo "  init (B)  ${INIT_CKPT}"
    echo "  teacher   ${TEACHER_CKPT}"
    cmd+=(--fsr --fsr_loss_scale "${FSR_SCALE}"
          --ckpt_path "${INIT_CKPT}" --teacher_ckpt "${TEACHER_CKPT}"
          --exp "${EXP:-nmrf_eth_fsrC_onB}")
    [[ "${FREEZE}" == "1" ]] && cmd+=(--freeze_backbone)
    ;;
  *)
    echo "GROUP must be B or C (A is test-only)"; exit 1
    ;;
esac

"${cmd[@]}"
