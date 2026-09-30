#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from typing import List, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from fsr_helpers import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    apply_fsr_config,
    clone_batch,
    horizon_cfg,
    load_state_dict_flexible,
    official_ckpt,
    unwrap_pred,
)

ETHUCY = ("eth", "hotel", "univ", "zara1", "zara2")
ALL_DATASETS = ETHUCY + ("nba", "sdd", "jrdb")


def apply_missing_locf(
    past: torch.Tensor,
    missing_ratio: float,
    seed: int,
    batch_idx: int,
) -> torch.Tensor:
    """past: [P, T, 2] or [B, N, T, 2]. t=0 always kept; LOCF fill."""
    if missing_ratio is None or float(missing_ratio) <= 0:
        return past
    x = past.clone()
    t_len = x.shape[-2]
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed) + int(batch_idx) * 1009)
    keep = (
        torch.rand(x.shape[:-1], generator=gen, device=x.device, dtype=x.dtype) > float(missing_ratio)
    ).to(dtype=x.dtype).unsqueeze(-1)
    keep[..., 0, :] = 1.0
    x[..., 0, :] = past[..., 0, :]
    for t in range(1, t_len):
        k = keep[..., t, :]
        x[..., t, :] = k * past[..., t, :] + (1.0 - k) * x[..., t - 1, :]
    return x


def parse_ratios(text: str) -> List[float]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    if not out:
        raise ValueError("missing_ratios is empty")
    return out


def resolve_ckpt(
    dataset: str,
    group: str,
    results_root: str,
    train_noise: float,
    fsr_scale: float,
    freeze: bool,
    ckpt_arg: str,
) -> str:
    if ckpt_arg:
        return ckpt_arg
    group = group.upper()
    ntr = "{:g}".format(float(train_noise))
    fsc = "{:g}".format(float(fsr_scale))
    if group == "A":
        return official_ckpt(dataset)
    if dataset == "nba":
        if group == "B":
            return os.path.join(
                results_root, "nba_nmrf_nba_noiseB_ntr{}".format(ntr), "nba_ckpt_best.pth"
            )
        preferred = os.path.join(
            results_root,
            "nba_nmrf_nba_fsrC_onB_FSR_frz_id0_mb2_fsc0.2_ntr{}".format(ntr),
            "nba_ckpt_best.pth",
        )
        if os.path.isfile(preferred):
            return preferred
        hits = sorted(
            glob.glob(os.path.join(results_root, "nba_nmrf_nba_fsrC_onB_*/nba_ckpt_best.pth"))
        )
        if hits:
            return hits[-1]
        hits = sorted(
            glob.glob(os.path.join(results_root, "nba_nmrf_nba_fsrC_sem_FSR_*/nba_ckpt_best.pth"))
        )
        return hits[-1] if hits else preferred
    if dataset == "sdd":
        if group == "B":
            return os.path.join(
                results_root, "sdd_nmrf_sdd_noiseB_ntr{}".format(ntr), "sdd_ckpt_best.pth"
            )
        preferred = os.path.join(
            results_root,
            "sdd_nmrf_sdd_fsrC_onB_FSR_frz_mb2_fsc0.2_ntr{}".format(ntr),
            "sdd_ckpt_best.pth",
        )
        if os.path.isfile(preferred):
            return preferred
        exp = "nmrf_sdd_fsrC_onB"
        tags = []
        base = "FSR"
        if freeze:
            base += "_frz"
        if fsc != "1":
            base += "_fsc{}".format(fsc)
        tags.append("{}_ntr{}".format(base, ntr))
        tags.append("FSR_frz_mb2_fsc{}_ntr{}".format(fsc, ntr))
        tags.append("FSR_id0_fsc{}_ntr{}".format(fsc, ntr))
        tags.append("FSR_fsc{}_ntr{}".format(fsc, ntr))
        cand = ""
        for tag in tags:
            cand = os.path.join(
                results_root, "sdd_{}_{}".format(exp, tag), "sdd_ckpt_best.pth"
            )
            if os.path.isfile(cand):
                return cand
        hits = sorted(
            glob.glob(os.path.join(results_root, "sdd_{}_*/sdd_ckpt_best.pth".format(exp)))
        )
        return hits[-1] if hits else cand
    if dataset == "jrdb":
        if group == "B":
            return os.path.join(
                results_root, "jrdb_nmrf_jrdb_noiseB_ntr{}".format(ntr), "jrdb_ckpt_best.pth"
            )
        preferred = os.path.join(
            results_root,
            "jrdb_nmrf_jrdb_fsrC_onB_FSR_frz_mb2_fsc0.2_ntr{}".format(ntr),
            "jrdb_ckpt_best.pth",
        )
        if os.path.isfile(preferred):
            return preferred
        exp = "nmrf_jrdb_fsrC_onB"
        tags = []
        base = "FSR"
        if freeze:
            base += "_frz"
        if fsc != "1":
            base += "_fsc{}".format(fsc)
        tags.append("{}_ntr{}".format(base, ntr))
        tags.append("FSR_frz_mb2_fsc{}_ntr{}".format(fsc, ntr))
        tags.append("FSR_id0_fsc{}_ntr{}".format(fsc, ntr))
        tags.append("FSR_fsc{}_ntr{}".format(fsc, ntr))
        cand = ""
        for tag in tags:
            cand = os.path.join(
                results_root, "jrdb_{}_{}".format(exp, tag), "jrdb_ckpt_best.pth"
            )
            if os.path.isfile(cand):
                return cand
        hits = sorted(
            glob.glob(os.path.join(results_root, "jrdb_{}_*/jrdb_ckpt_best.pth".format(exp)))
        )
        return hits[-1] if hits else cand
    if group == "B":
        return os.path.join(
            results_root,
            "{}_nmrf_eth_noiseB_ntr{}".format(dataset, ntr),
            "{}_ckpt_best.pth".format(dataset),
        )
    exp = "nmrf_eth_fsrC_onB"
    tags = []
    base = "FSR"
    if freeze:
        base += "_frz"
    if fsc != "1":
        base += "_fsc{}".format(fsc)
    tags.append("{}_ntr{}".format(base, ntr))
    tags.append("FSR_id0_fsc{}_ntr{}".format(fsc, ntr))
    tags.append("FSR_fsc{}_ntr{}".format(fsc, ntr))
    for tag in tags:
        cand = os.path.join(
            results_root,
            "{}_{}_{}".format(dataset, exp, tag),
            "{}_ckpt_best.pth".format(dataset),
        )
        if os.path.isfile(cand):
            return cand
    hits = sorted(
        glob.glob(
            os.path.join(results_root, "{}_{}_*/{}_ckpt_best.pth".format(dataset, exp, dataset))
        )
    )
    return hits[-1] if hits else cand


def build_config(args, dataset: str, group: str, ckpt: str):
    cfg = argparse.Namespace(
        dataset=dataset,
        train=False,
        log="missing_eval",
        use_sampler=False,
        epoch=150,
        gpu=str(args.gpu),
        fsr=group.upper() == "C",
        fsr_sem=False,
        ckpt_path=ckpt,
        teacher_ckpt="",
        exp="missing_eval",
        results_root=args.results_root,
        fsr_tau=0.1,
        freeze_backbone=False,
        freeze_encoder=False,
        load_full=False,
        num_epochs=-1,
        train_noise_sigma=0.0,
        eval_noise_sigma=0.0,
        smooth_past_deg=-1,
        noise_seed=int(args.seed),
    )
    return apply_fsr_config(cfg)


def make_trainer(cfg):
    if cfg.dataset in ETHUCY:
        from sample_mrf.sample_mrf_ethucy import Trainer
    elif cfg.dataset == "nba":
        from sample_mrf.sample_mrf_nba import Trainer
    elif cfg.dataset == "sdd":
        from sample_mrf.sample_mrf_sdd import Trainer
    elif cfg.dataset == "jrdb":
        from sample_mrf.sample_mrf_jrdb import Trainer
    else:
        raise ValueError("unsupported dataset: {}".format(cfg.dataset))
    os.makedirs("./logs", exist_ok=True)
    return Trainer(cfg)


@torch.no_grad()
def eval_missing(trainer, missing_ratio: float, seed: int) -> Tuple[dict, int]:
    trainer.mrf_predictor.eval()
    horizon = horizon_cfg(trainer.config.dataset)
    if horizon is not None:
        performance = {"FDE": [0, 0, 0, 0], "ADE": [0, 0, 0, 0]}
    else:
        performance = {"FDE": 0, "ADE": 0}
    samples = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for i, data in enumerate(trainer.test_loader):
        data_in = clone_batch(data)
        data_in["pre_motion_3D"] = apply_missing_locf(
            data_in["pre_motion_3D"], missing_ratio, seed=seed, batch_idx=i
        )
        proc = trainer.data_preprocess(data_in)
        fut = proc["fut_traj"].unsqueeze(0).repeat(trainer.N, 1, 1, 1)
        pred = unwrap_pred(trainer.mrf_predictor(proc, trainer.N, vae_train=False, noise=True))
        distances = torch.norm(fut - pred, dim=-1) * trainer.traj_scale
        if trainer.config.dataset == "sdd" and "peds_scales" in proc:
            distances = distances * proc["peds_scales"].to(distances.device).unsqueeze(0).unsqueeze(-1)
        if horizon is not None:
            step = horizon["step"]
            for time_i in range(1, 5):
                ade = distances[:, :, : step * time_i].mean(dim=-1).min(dim=0)[0].sum()
                fde = distances[:, :, step * time_i - 1].min(dim=0)[0].sum()
                performance["ADE"][time_i - 1] += ade.item()
                performance["FDE"][time_i - 1] += fde.item()
        else:
            performance["ADE"] += distances.mean(dim=-1).min(dim=0)[0].sum().item()
            performance["FDE"] += distances[:, :, -1].min(dim=0)[0].sum().item()
        samples += distances.shape[1]
    return performance, samples


def log_result(trainer, performance, samples, prefix: str) -> None:
    horizon = horizon_cfg(trainer.config.dataset)
    if horizon is not None:
        for t, sec in enumerate(horizon["secs"]):
            trainer.print_log(
                "{}--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}".format(
                    prefix,
                    sec,
                    performance["ADE"][t] / samples,
                    sec,
                    performance["FDE"][t] / samples,
                ),
                trainer.log,
            )
    else:
        trainer.print_log(
            "{}--ADE(4.8s): {:.4f}\t--FDE(4.8s): {:.4f}".format(
                prefix, performance["ADE"] / samples, performance["FDE"] / samples
            ),
            trainer.log,
        )


def parse_groups(raw: str) -> List[str]:
    g = str(raw).strip()
    if g.upper() in ("A", "B", "C"):
        return [g.upper()]
    if g.lower() in ("0", "all", "*", ""):
        print(
            "[WARN] --group={!r} is not A/B/C (AutoDL often sets GROUP=0); running A, B, C".format(
                raw
            )
        )
        return ["A", "B", "C"]
    raise ValueError("--group must be A, B, C, or all; got {!r}".format(raw))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NMRF A/B/C missing-frame (LOCF) eval")
    p.add_argument("--dataset", required=True, help="eth|hotel|univ|zara1|zara2|nba|sdd|jrdb")
    p.add_argument(
        "--group",
        required=True,
        help="A, B, C, or all (0 from AutoDL GROUP env is treated as all)",
    )
    p.add_argument("--missing_ratios", default="0.1,0.2,0.3")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", default="0")
    p.add_argument("--ckpt", default="", help="Override default A/B/C checkpoint")
    p.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--train_noise", type=float, default=0.1)
    p.add_argument("--fsr_scale", type=float, default=None, help="Default 0.2 on ETH, 1.0 on NBA")
    p.add_argument("--freeze", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset = str(args.dataset).lower()
    if dataset not in ALL_DATASETS:
        raise ValueError("dataset must be one of {}, got {}".format(ALL_DATASETS, dataset))
    groups = parse_groups(args.group)
    ratios = parse_ratios(args.missing_ratios)
    fsr_scale = args.fsr_scale
    if fsr_scale is None:
        fsr_scale = 1.0 if dataset == "nba" else 0.2

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    for group in groups:
        ckpt = resolve_ckpt(
            dataset,
            group,
            args.results_root,
            args.train_noise,
            float(fsr_scale),
            bool(args.freeze),
            args.ckpt if len(groups) == 1 else "",
        )
        if not os.path.isfile(ckpt):
            raise FileNotFoundError("ckpt not found for group {}: {}".format(group, ckpt))

        cfg = build_config(args, dataset, group, ckpt)
        trainer = make_trainer(cfg)
        load_state_dict_flexible(
            trainer.mrf_predictor, ckpt, strict=not bool(cfg.use_fsr)
        )
        trainer.print_log(
            "[INFO] missing-eval dataset={} group={} ckpt={}".format(dataset, group, ckpt),
            trainer.log,
        )
        trainer.print_log(
            "[INFO] ratios={} seed={} C-smooth=off LOCF keep-t0".format(ratios, args.seed),
            trainer.log,
        )
        for ratio in ratios:
            performance, samples = eval_missing(trainer, ratio, int(args.seed))
            log_result(
                trainer,
                performance,
                samples,
                prefix="[missing={:g} group={}] ".format(ratio, group),
            )
        try:
            trainer.log.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
