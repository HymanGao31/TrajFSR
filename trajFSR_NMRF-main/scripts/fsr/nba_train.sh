#!/usr/bin/env bash
# NBA train.
#   GROUP=B bash scripts/fsr/nba_train.sh
#   GROUP=C bash scripts/fsr/nba_train.sh   # C-on-B
set -euo pipefail
cd "$(dirname "$0")/../.."

GROUP="${GROUP:-C}"             # B | C
GPU="${GPU:-0}"
TRAIN_NOISE="${TRAIN_NOISE:-0.1}"
EPOCHS="${EPOCHS:-100}"
FREEZE="${FREEZE:-0}"
FSR_SCALE="${FSR_SCALE:-1.0}"
M_LO="${M_LO:-0.05}"
M_HI="${M_HI:-0.20}"
DELTA="${DELTA:-0.02}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/nmrfFSR}"
INIT_CKPT="${INIT_CKPT:-}"
EXP="${EXP:-}"

cmd=(python main.py --dataset nba --train --gpu "${GPU}"
     --load_full
     --train_noise_sigma "${TRAIN_NOISE}" --eval_noise_sigma 0.2
     --results_root "${RESULTS_ROOT}")
[[ "${EPOCHS}" != "-1" ]] && cmd+=(--num_epochs "${EPOCHS}")

case "${GROUP}" in
  B)
    echo "===== TRAIN NBA B | ntr=${TRAIN_NOISE} ====="
    cmd+=(--exp "${EXP:-nmrf_nba_noiseB}")
    [[ -n "${INIT_CKPT}" ]] && cmd+=(--ckpt_path "${INIT_CKPT}")
    ;;
  C)
    if [[ -z "${INIT_CKPT}" ]]; then
      INIT_CKPT="${RESULTS_ROOT}/nba_nmrf_nba_noiseB_ntr${TRAIN_NOISE}/nba_ckpt_best.pth"
    fi
    [[ -f "${INIT_CKPT}" ]] || { echo "missing Group B ckpt: ${INIT_CKPT}"; echo "train B first: GROUP=B bash scripts/fsr/nba_train.sh"; exit 1; }
    echo "===== TRAIN NBA C-sem-band | ntr=${TRAIN_NOISE} | freeze=${FREEZE} | fsc=${FSR_SCALE} ====="
    echo "  init (B)  ${INIT_CKPT}"
    echo "  FSR sem   Z+ minADE; Z- band [Z++${M_LO}, Z++${M_HI}]; Z* beat Z+ by ${DELTA}"
    cmd+=(--fsr --fsr_sem --fsr_loss_scale "${FSR_SCALE}"
          --fsr_sem_m_lo "${M_LO}" --fsr_sem_m_hi "${M_HI}" --fsr_sem_delta "${DELTA}"
          --ckpt_path "${INIT_CKPT}"
          --exp "${EXP:-nmrf_nba_fsrC_sem}")
    if [[ "${FREEZE}" == "enc" ]]; then
      cmd+=(--freeze_encoder)
    elif [[ "${FREEZE}" == "1" ]]; then
      cmd+=(--freeze_backbone)
    fi
    ;;
  *)
    echo "GROUP must be B or C (A is test-only)"; exit 1
    ;;
esac

"${cmd[@]}"
