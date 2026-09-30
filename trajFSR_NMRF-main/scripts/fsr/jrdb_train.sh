#!/usr/bin/env bash
# JRDB train. Group A has no released weights here, so train official NMRF first:
#   FSR_GROUP=A bash scripts/fsr/jrdb_train.sh
# Then plugin B → C (ETH/SDD-style C-on-B):
#   FSR_GROUP=B bash scripts/fsr/jrdb_train.sh
#   FSR_GROUP=C bash scripts/fsr/jrdb_train.sh
# If C mask→1 (plugin off), rerun the SDD-proven anti-collapse combo:
#   FSR_GROUP=C FREEZE=1 MASK_BAL=2 bash scripts/fsr/jrdb_train.sh
# Fallback (keeps mask~0.5 via decoder ADE, like NBA semantic C):
#   FSR_GROUP=C SEM=1 bash scripts/fsr/jrdb_train.sh
#
# A stages (official yaml: VAE 200, sampler 50; last sampler save is model_0049.p):
#   FSR_GROUP=A bash scripts/fsr/jrdb_train.sh              # VAE then sampler
#   FSR_GROUP=A A_STAGE=vae bash scripts/fsr/jrdb_train.sh  # VAE only
#   FSR_GROUP=A A_STAGE=sampler VAE_EPOCH=199 bash scripts/fsr/jrdb_train.sh
# Needs processed_datasets/jrdb/trajectories_jrdb_{train,val}.pkl
set -euo pipefail
cd "$(dirname "$0")/../.."

# AutoDL sets GROUP=0. Prefer FSR_GROUP.
if [[ -n "${FSR_GROUP:-}" ]]; then
  GROUP="${FSR_GROUP}"
else
  case "${GROUP:-}" in
    A|a|B|b|C|c) GROUP="${GROUP}" ;;
    "") GROUP="A" ;;
    *)
      echo "[ERR] AutoDL GROUP='${GROUP}' is not A/B/C. Use FSR_GROUP=A|B|C"
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
A_LOG="${A_LOG:-default}"
A_STAGE="${A_STAGE:-all}"
VAE_EPOCH="${VAE_EPOCH:-}"
A_CKPT="results/jrdb/${A_LOG}/model_0049.p"

latest_jrdb_ckpt_epoch() {
  local dir="results/jrdb/${A_LOG}"
  local latest="" f b e
  shopt -s nullglob
  for f in "${dir}"/model_*.p; do
    b="$(basename "${f}" .p)"
    e="${b#model_}"
    e=$((10#${e}))
    if [[ -z "${latest}" || "${e}" -gt "${latest}" ]]; then
      latest="${e}"
    fi
  done
  shopt -u nullglob
  echo "${latest}"
}

case "${GROUP}" in
  A)
    A_STAGE="$(echo "${A_STAGE}" | tr '[:upper:]' '[:lower:]')"
    mkdir -p "./logs" "./results/jrdb/${A_LOG}"
    if [[ "${A_STAGE}" == "all" || "${A_STAGE}" == "vae" || "${A_STAGE}" == "1" ]]; then
      echo "===== TRAIN JRDB A stage1 CVAE | log=${A_LOG} ====="
      python main.py --dataset jrdb --train --log "${A_LOG}" --gpu "${GPU}"
    fi
    if [[ "${A_STAGE}" == "all" || "${A_STAGE}" == "sampler" || "${A_STAGE}" == "2" ]]; then
      if [[ -z "${VAE_EPOCH}" ]]; then
        VAE_EPOCH="$(latest_jrdb_ckpt_epoch)"
      fi
      [[ -n "${VAE_EPOCH}" ]] || {
        echo "missing VAE ckpt under results/jrdb/${A_LOG}/"
        echo "run stage1 first: FSR_GROUP=A A_STAGE=vae bash scripts/fsr/jrdb_train.sh"
        exit 1
      }
      echo "===== TRAIN JRDB A stage2 sampler | load VAE epoch=${VAE_EPOCH} | log=${A_LOG} ====="
      python main.py --dataset jrdb --train --log "${A_LOG}" --gpu "${GPU}" \
        --use_sampler --epoch "${VAE_EPOCH}"
      [[ -f "${A_CKPT}" ]] || {
        echo "[WARN] expected sampler ckpt missing: ${A_CKPT}"
        echo "sampler saves every 5 epochs; last of 50 is model_0049.p"
        exit 1
      }
      echo "[INFO] JRDB A ready: ${A_CKPT}"
    fi
    exit 0
    ;;
  B)
    if [[ -z "${INIT_CKPT}" ]]; then
      [[ -f "${A_CKPT}" ]] || {
        echo "missing Group A ckpt: ${A_CKPT}"
        echo "train A first: FSR_GROUP=A bash scripts/fsr/jrdb_train.sh"
        exit 1
      }
    fi
    echo "===== TRAIN JRDB B | ntr=${TRAIN_NOISE} ====="
    cmd=(python main.py --dataset jrdb --train --gpu "${GPU}"
         --load_full
         --train_noise_sigma "${TRAIN_NOISE}" --eval_noise_sigma 0.2
         --results_root "${RESULTS_ROOT}"
         --exp "${EXP:-nmrf_jrdb_noiseB}")
    [[ "${EPOCHS}" != "-1" ]] && cmd+=(--num_epochs "${EPOCHS}")
    [[ -n "${INIT_CKPT}" ]] && cmd+=(--ckpt_path "${INIT_CKPT}")
    ;;
  C)
    if [[ -z "${INIT_CKPT}" ]]; then
      INIT_CKPT="${RESULTS_ROOT}/jrdb_nmrf_jrdb_noiseB_ntr${TRAIN_NOISE}/jrdb_ckpt_best.pth"
    fi
    if [[ -z "${TEACHER_CKPT}" ]]; then
      TEACHER_CKPT="${A_CKPT}"
    fi
    [[ -f "${INIT_CKPT}" ]] || { echo "missing Group B ckpt: ${INIT_CKPT}"; echo "train B first: FSR_GROUP=B bash scripts/fsr/jrdb_train.sh"; exit 1; }
    [[ "${SEM}" != "1" && ! -f "${TEACHER_CKPT}" ]] && {
      echo "missing teacher (Group A): ${TEACHER_CKPT}"
      echo "train A first: FSR_GROUP=A bash scripts/fsr/jrdb_train.sh"
      exit 1
    }
    echo "===== TRAIN JRDB C-on-B | ntr=${TRAIN_NOISE} | freeze=${FREEZE} | fsc=${FSR_SCALE} | mb=${MASK_BAL} | sem=${SEM} ====="
    echo "  init (B)  ${INIT_CKPT}"
    cmd=(python main.py --dataset jrdb --train --gpu "${GPU}"
         --load_full
         --train_noise_sigma "${TRAIN_NOISE}" --eval_noise_sigma 0.2
         --results_root "${RESULTS_ROOT}"
         --fsr --fsr_loss_scale "${FSR_SCALE}" --ckpt_path "${INIT_CKPT}")
    [[ "${EPOCHS}" != "-1" ]] && cmd+=(--num_epochs "${EPOCHS}")
    if [[ "${SEM}" == "1" ]]; then
      echo "  FSR sem   Z+ minADE; Z- band; Z* beat Z+"
      cmd+=(--fsr_sem --exp "${EXP:-nmrf_jrdb_fsrC_sem}")
    else
      echo "  teacher   ${TEACHER_CKPT}"
      cmd+=(--teacher_ckpt "${TEACHER_CKPT}" --exp "${EXP:-nmrf_jrdb_fsrC_onB}")
      [[ "${MASK_BAL}" != "0" ]] && cmd+=(--lam_mask_bal "${MASK_BAL}")
    fi
    if [[ "${FREEZE}" == "enc" ]]; then
      cmd+=(--freeze_encoder)
    elif [[ "${FREEZE}" == "1" ]]; then
      cmd+=(--freeze_backbone)
    fi
    ;;
  *)
    echo "FSR_GROUP must be A, B, or C"
    exit 1
    ;;
esac

"${cmd[@]}"
