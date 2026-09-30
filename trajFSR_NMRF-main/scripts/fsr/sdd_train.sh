#!/usr/bin/env bash
# SDD train (plugin). Official NMRF SDD (Group A) is already:
#   python main.py --dataset sdd --train --log default
#   python main.py --dataset sdd --train --log default --use_sampler --epoch 80
# Plugin (ETH-style C-on-B, not NBA semantic C):
#   FSR_GROUP=B bash scripts/fsr/sdd_train.sh
#   FSR_GROUP=C bash scripts/fsr/sdd_train.sh
# If C mask→1 (plugin off), rerun the NBA-proven anti-collapse combo:
#   FSR_GROUP=C FREEZE=1 MASK_BAL=2 bash scripts/fsr/sdd_train.sh
# Fallback (keeps mask~0.5 via decoder ADE, like NBA semantic C):
#   FSR_GROUP=C SEM=1 bash scripts/fsr/sdd_train.sh
# Needs processed_datasets/sdd_{train,test}.pkl at repo root (see preprocess/prepare_sdd.py).
set -euo pipefail
cd "$(dirname "$0")/../.."

# AutoDL sets GROUP=0. Prefer FSR_GROUP.
if [[ -n "${FSR_GROUP:-}" ]]; then
  GROUP="${FSR_GROUP}"
else
  case "${GROUP:-}" in
    B|b|C|c) GROUP="${GROUP}" ;;
    "") GROUP="C" ;;
    *)
      echo "[ERR] AutoDL GROUP='${GROUP}' is not B/C. Use FSR_GROUP=B or FSR_GROUP=C"
      exit 1
      ;;
  esac
fi
GROUP="$(echo "${GROUP}" | tr '[:lower:]' '[:upper:]')"

GPU="${GPU:-0}"
TRAIN_NOISE="${TRAIN_NOISE:-0.1}"
EPOCHS="${EPOCHS:-100}"
FREEZE="${FREEZE:-0}"
FSR_SCALE="${FSR_SCALE:-0.2}"
MASK_BAL="${MASK_BAL:-0}"
SEM="${SEM:-0}"
RESULTS_ROOT="${RESULTS_ROOT:-/root/autodl-tmp/nmrfFSR}"
INIT_CKPT="${INIT_CKPT:-}"
TEACHER_CKPT="${TEACHER_CKPT:-}"
EXP="${EXP:-}"

cmd=(python main.py --dataset sdd --train --gpu "${GPU}"
     --load_full
     --train_noise_sigma "${TRAIN_NOISE}" --eval_noise_sigma 0.2
     --results_root "${RESULTS_ROOT}")
[[ "${EPOCHS}" != "-1" ]] && cmd+=(--num_epochs "${EPOCHS}")

case "${GROUP}" in
  B)
    echo "===== TRAIN SDD B | ntr=${TRAIN_NOISE} ====="
    cmd+=(--exp "${EXP:-nmrf_sdd_noiseB}")
    [[ -n "${INIT_CKPT}" ]] && cmd+=(--ckpt_path "${INIT_CKPT}")
    ;;
  C)
    if [[ -z "${INIT_CKPT}" ]]; then
      INIT_CKPT="${RESULTS_ROOT}/sdd_nmrf_sdd_noiseB_ntr${TRAIN_NOISE}/sdd_ckpt_best.pth"
    fi
    if [[ -z "${TEACHER_CKPT}" ]]; then
      TEACHER_CKPT="results/sdd/default/model_0080.p"
    fi
    [[ -f "${INIT_CKPT}" ]] || { echo "missing Group B ckpt: ${INIT_CKPT}"; echo "train B first: FSR_GROUP=B bash scripts/fsr/sdd_train.sh"; exit 1; }
    [[ "${SEM}" != "1" && ! -f "${TEACHER_CKPT}" ]] && { echo "missing teacher: ${TEACHER_CKPT}"; exit 1; }
    echo "===== TRAIN SDD C-on-B | ntr=${TRAIN_NOISE} | freeze=${FREEZE} | fsc=${FSR_SCALE} | mb=${MASK_BAL} | sem=${SEM} ====="
    echo "  init (B)  ${INIT_CKPT}"
    cmd+=(--fsr --fsr_loss_scale "${FSR_SCALE}" --ckpt_path "${INIT_CKPT}")
    if [[ "${SEM}" == "1" ]]; then
      echo "  FSR sem   Z+ minADE; Z- band; Z* beat Z+"
      cmd+=(--fsr_sem --exp "${EXP:-nmrf_sdd_fsrC_sem}")
    else
      echo "  teacher   ${TEACHER_CKPT}"
      cmd+=(--teacher_ckpt "${TEACHER_CKPT}" --exp "${EXP:-nmrf_sdd_fsrC_onB}")
      [[ "${MASK_BAL}" != "0" ]] && cmd+=(--lam_mask_bal "${MASK_BAL}")
    fi
    if [[ "${FREEZE}" == "enc" ]]; then
      cmd+=(--freeze_encoder)
    elif [[ "${FREEZE}" == "1" ]]; then
      cmd+=(--freeze_backbone)
    fi
    ;;
  *)
    echo "FSR_GROUP must be B or C (A is official NMRF: python main.py --dataset sdd --train)"
    exit 1
    ;;
esac

"${cmd[@]}"
