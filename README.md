# NMRF + TrajFSR (plugin)

Pluginized **TrajFSR** on NMRF (ICLR 2025). Official two-stage CVAE→sampler training is unchanged; Groups B/C are a **single-loop fine-tune of the full pretrained net** (encoder + dynamics + sampler) with sampler inference (`vae_train=False, noise=True`).

```text
past xy (6-d: abs / rel / vel)
  → History Encoder (Conv1d+GRU+attn) → [P, 256]
  → TrajFSR (optional)  → r_feat + rec_feat
  → Update Decoder / learned sampler
```

- **Teacher**: frozen official NMRF pretrained (no FSR) under `results/{ds}/default/model_XXXX.p`
- **Student**: same backbone + newly initialized FSR
- **A**: frozen official, test only
- **B**: noise fine-tune, no FSR
- **C**: noise + FSR; **+smooth** is test-time polynomial degree 2 on the same ckpt
- Datasets: **ETH/UCY, NBA, SDD, JRDB**

Train-time smooth is off. Test scripts always sweep noise `0,0.1,...,0.5` twice: `smooth=-1` then `smooth=2`.

ETH/UCY Group C uses `LAM_ID=0.5` and `FSR_SCALE=0.5` so the Gumbel mask does not saturate at 1. NBA Group C stays `0.5 / 1.0`.

## Scripts (`scripts/fsr/`)

Edit the top `# =====  =====` block, then run. Official pretrained weights must already be at `results/{subset}/default/model_{epoch}.p` (eth 47, hotel 44, univ 15, zara1 47, zara2 55, nba 14, sdd 80, jrdb 49).

| Dataset | Train | Test |
|---------|-------|------|
| ETH/UCY | `eth_train.sh` (`GROUP=B\|C`, `SUBSET=...`, `EPOCHS=60`) | `eth_test.sh` (`GROUP=A\|B\|C`) |
| NBA | `nba_train.sh` (`EPOCHS=50`) | `nba_test.sh` |
| SDD | `sdd_train.sh` (`FSR_GROUP=B\|C`) | `sdd_test.sh` |
| JRDB | `jrdb_train.sh` (`FSR_GROUP=A\|B\|C`) | `jrdb_test.sh` |

```bash
cd /path/to/trajFSR_NMRF

# ETH hotel, Group C
# in eth_train.sh / eth_test.sh set: SUBSET=hotel  GROUP=C
bash scripts/fsr/eth_train.sh
bash scripts/fsr/eth_test.sh

bash scripts/fsr/nba_train.sh
bash scripts/fsr/nba_test.sh
```

Original NMRF CLI is unchanged when FSR/noise flags are off, for example:

```bash
python main.py --dataset eth --log default --use_sampler --epoch 47
```

## Weights

```text
/root/autodl-tmp/nmrfFSR/
  {subset}_nmrf_eth_noiseB_ntr0.1/{subset}_ckpt_best.pth
  {subset}_nmrf_eth_fsrC_FSR_fsc0.5_ntr0.1/{subset}_ckpt_best.pth
  nba_nmrf_nba_noiseB_ntr0.1/nba_ckpt_best.pth
  nba_nmrf_nba_fsrC_FSR_ntr0.1/nba_ckpt_best.pth
  jrdb_nmrf_jrdb_noiseB_ntr0.1/jrdb_ckpt_best.pth
  jrdb_nmrf_jrdb_fsrC_onB_FSR_fsc0.2_ntr0.1/jrdb_ckpt_best.pth

results/{eth,hotel,univ,zara1,zara2,nba,sdd,jrdb}/default/model_XXXX.p
```

Empty `CKPT=` in a test script uses the default path for that `GROUP`. ETH/UCY B/C select **val ADE**; NBA selects **ADE@4s**; JRDB selects **ADE@4.8s**.

`FREEZE=1` trains FSR only. Default `FREEZE=0` fine-tunes the full net.

## Flags

| Flag | Meaning |
|------|---------|
| `--fsr` | Insert TrajFSR after the History Encoder (256-d) |
| `--load_full` | Init student from official / `--ckpt_path` |
| `--freeze_backbone` | Train only FSR (`FREEZE=1` in train scripts) |
| `--train_noise_sigma` / `--eval_noise_sigma` | Gaussian on past abs xy, then rebuild rel/vel |
| `--smooth_past_deg` | Test-time poly degree (`2`; `-1` off) |
| `--lam_id` / `--fsr_loss_scale` | Identity weight / aux-loss scale (ETH C: `0.5` / `0.5`) |



GROUP=C SUBSET=eth bash scripts/fsr/eth_train.sh
GROUP=B SUBSET=hotel GPU=0 EPOCHS=60 bash scripts/fsr/eth_train.sh


GROUP=B SUBSET=eth bash scripts/fsr/eth_test.sh
GROUP=C SUBSET=eth bash scripts/fsr/eth_test.sh
GROUP=A SUBSET=eth bash scripts/fsr/eth_test.sh

GROUP=C GPU=0 EPOCHS=50 bash scripts/fsr/nba_train.sh

GROUP=B bash scripts/fsr/nba_test.sh
GROUP=C bash scripts/fsr/nba_test.sh


# Group A
python main.py --dataset sdd --train --log default
python main.py --dataset sdd --train --log default --use_sampler --epoch 80
#  B → C
FSR_GROUP=B bash scripts/fsr/sdd_train.sh
FSR_GROUP=C FREEZE=1 MASK_BAL=2 bash scripts/fsr/sdd_train.sh
# （σ=0..0.5，smooth=-1 \ 2）
FSR_GROUP=A bash scripts/fsr/sdd_test.sh
FSR_GROUP=B bash scripts/fsr/sdd_test.sh
FSR_GROUP=C FREEZE=1 MASK_BAL=2 bash scripts/fsr/sdd_test.sh
# miss LOCF
DATASET=sdd FSR_GROUP=C bash scripts/fsr/test_missing.sh



FSR_GROUP=A bash scripts/fsr/sdd_natra_test.sh
FSR_GROUP=B bash scripts/fsr/sdd_natra_test.sh
FSR_GROUP=C bash scripts/fsr/sdd_natra_test.sh


# JRDB  A（CVAE 200 → sampler 50 → results/jrdb/default/model_0049.p）
FSR_GROUP=A bash scripts/fsr/jrdb_train.sh
#  B → C（C-on-B）
FSR_GROUP=B bash scripts/fsr/jrdb_train.sh
FSR_GROUP=C bash scripts/fsr/jrdb_train.sh

# （σ=0..0.5，smooth=-1\2）
FSR_GROUP=A bash scripts/fsr/jrdb_test.sh
FSR_GROUP=B bash scripts/fsr/jrdb_test.sh
FSR_GROUP=C bash scripts/fsr/jrdb_test.sh
# miss LOCF
DATASET=jrdb FSR_GROUP=C bash scripts/fsr/test_missing.sh


# NATRA
FSR_GROUP=A bash scripts/fsr/jrdb_natra_test.sh
FSR_GROUP=B bash scripts/fsr/jrdb_natra_test.sh
FSR_GROUP=C bash scripts/fsr/jrdb_natra_test.sh

