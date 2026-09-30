"""Single-loop plugin fine-tune / eval for NMRF + TrajFSR.

Official two-stage CVAE→sampler training is left unchanged. This path loads the
full pretrained net and fine-tunes with sampler inference (vae_train=False, noise=True).
"""

from __future__ import annotations

import os
import time

import torch

from fsr_helpers import (
    apply_param_freeze,
    build_save_dir,
    build_teacher,
    clone_batch,
    compute_fsr_loss,
    corrupt_batch,
    horizon_cfg,
    is_plugin_finetune,
    load_state_dict_flexible,
    official_ckpt,
    should_use_plugin_test,
    unwrap_pred,
)


def _temporal_reweight(trainer):
    div = 10.0 if trainer.config.dataset == "nba" else 6.0
    w = [(trainer.fut_step + 1) - i for i in range(1, trainer.fut_step + 1)]
    return torch.FloatTensor(w).cuda().unsqueeze(0).unsqueeze(0) / div


def _sampler_forward(model, proc, n_samples):
    return unwrap_pred(model(proc, n_samples, vae_train=False, noise=True))


def _ade_loss(pred, fut, reweight):
    return ((pred - fut.unsqueeze(0)).norm(p=2, dim=-1) * reweight).mean(dim=-1).min(dim=0)[0].mean()


def configure_plugin_training(trainer):
    config = trainer.config
    if not is_plugin_finetune(config):
        return

    log_tag = getattr(config, "exp", "") or (
        config.log if getattr(config, "log", "") and config.log != "default" else "nmrf_plugin"
    )
    os.makedirs("./logs", exist_ok=True)
    try:
        trainer.log.close()
    except Exception:
        pass
    trainer.log = open(os.path.join("./logs/log_{}_{}.txt".format(config.dataset, log_tag)), "a+")

    ckpt = getattr(config, "ckpt_path", "") or official_ckpt(config.dataset, config.log or "default")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError("Plugin init checkpoint not found: {}".format(ckpt))

    epoch, missing, unexpected = load_state_dict_flexible(
        trainer.mrf_predictor, ckpt, strict=False
    )
    trainer.print_log("[INFO] Loaded student from {} (epoch={})".format(ckpt, epoch), trainer.log)
    if getattr(config, "use_fsr", False) and missing:
        fsr_miss = [k for k in missing if "fsr" in k]
        other = [k for k in missing if "fsr" not in k]
        trainer.print_log("[INFO] Missing FSR keys (expected): {}".format(len(fsr_miss)), trainer.log)
        if other:
            trainer.print_log("[WARN] Other missing keys: {}".format(list(other)[:8]), trainer.log)
    if unexpected:
        trainer.print_log("[WARN] Unexpected keys: {}".format(list(unexpected)[:8]), trainer.log)

    apply_param_freeze(
        trainer.mrf_predictor,
        freeze_backbone=bool(getattr(config, "freeze_backbone", False)),
        freeze_encoder=bool(getattr(config, "freeze_encoder", False)),
    )
    train_params = [p for p in trainer.mrf_predictor.parameters() if p.requires_grad]
    trainer.optimizer = torch.optim.AdamW(train_params, lr=trainer.hyper_config["lr"])
    trainer.scheduler_model = torch.optim.lr_scheduler.StepLR(
        trainer.optimizer,
        step_size=trainer.hyper_config["step_size"],
        gamma=trainer.hyper_config["gamma"],
    )

    trainer.save_dir = build_save_dir(config, config.dataset)
    trainer.print_log("[INFO] Save dir: {}".format(trainer.save_dir), trainer.log)

    trainer.teacher = None
    if getattr(config, "use_fsr", False) and not getattr(config, "fsr_sem", False):
        teacher_path = getattr(config, "teacher_ckpt", "") or ckpt
        if not os.path.isfile(teacher_path):
            raise FileNotFoundError("Teacher checkpoint not found: {}".format(teacher_path))
        trainer.teacher = build_teacher(
            config, teacher_path, trainer.fut_step, trainer.hyper_config, trainer.N
        )
        trainer.print_log("[INFO] Teacher loaded (no FSR) from {}".format(teacher_path), trainer.log)
    elif getattr(config, "fsr_sem", False):
        m_lo = float(getattr(config, "fsr_sem_m_lo", 0.05))
        m_hi = float(getattr(config, "fsr_sem_m_hi", 0.20))
        delta = float(getattr(config, "fsr_sem_delta", 0.02))
        trainer.print_log(
            "[INFO] FSR semantic loss: Z+ minADE; Z- band [Z++{:.3f}, Z++{:.3f}]; "
            "Z* must beat Z+ by {:.3f}; no teacher".format(m_lo, m_hi, delta),
            trainer.log,
        )


def _eval_noise_sigma(trainer, override=None):
    if override is not None:
        return float(override)
    return float(getattr(trainer.config, "eval_noise_sigma", 0.0) or 0.0)


def _eval_smooth(trainer, override=None):
    if override is not None:
        return int(override)
    return int(getattr(trainer.config, "smooth_past_deg", -1))


def eval_plugin(trainer, noise_sigma=None, smooth_deg=None):
    trainer.mrf_predictor.eval()
    horizon = horizon_cfg(trainer.config.dataset)
    if horizon is not None:
        import random
        import numpy as np

        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        performance = {"FDE": [0, 0, 0, 0], "ADE": [0, 0, 0, 0]}
    else:
        performance = {"FDE": 0, "ADE": 0}

    sigma = _eval_noise_sigma(trainer, noise_sigma)
    sm = _eval_smooth(trainer, smooth_deg)
    seed = int(getattr(trainer.config, "noise_seed", 42))
    protocol = str(getattr(trainer.config, "noise_protocol", "gaussian") or "gaussian")
    samples = 0

    with torch.no_grad():
        for i, data in enumerate(trainer.test_loader):
            data_in = corrupt_batch(
                data,
                noise_sigma=sigma,
                smooth_deg=sm,
                seed=seed,
                batch_idx=i,
                protocol=protocol,
                poisson_lam=float(getattr(trainer.config, "poisson_lam", 0.4)),
                mixed_gaussian_sigma=float(getattr(trainer.config, "mixed_gaussian_sigma", 0.2)),
                mixed_poisson_lam=float(getattr(trainer.config, "mixed_poisson_lam", 0.2)),
            )
            proc = trainer.data_preprocess(data_in)
            fut = proc["fut_traj"].unsqueeze(0).repeat(trainer.N, 1, 1, 1)
            pred = _sampler_forward(trainer.mrf_predictor, proc, trainer.N)
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


def _select_score(trainer, performance, samples):
    if horizon_cfg(trainer.config.dataset) is not None:
        return performance["ADE"][3] / samples
    return performance["ADE"] / samples


def _log_eval(trainer, performance, samples, prefix=""):
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


def fit_plugin(trainer):
    configure_plugin_training(trainer)
    config = trainer.config
    total_epoch = int(getattr(config, "num_epochs", -1) or -1)
    if total_epoch <= 0:
        total_epoch = int(trainer.hyper_config["num_epoch_sampler"])
    trainer.print_log("Plugin fine-tune epochs: {}".format(total_epoch), trainer.log)

    reweight = _temporal_reweight(trainer)
    best = 1e8
    save_dir = trainer.save_dir
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "{}_ckpt_best.pth".format(config.dataset))
    ntr = float(getattr(config, "train_noise_sigma", 0.0) or 0.0)
    seed = int(getattr(config, "noise_seed", 42))
    loader_len = len(trainer.train_loader)

    for epoch in range(total_epoch):
        trainer.mrf_predictor.train()
        loss_total, loss_rcn, loss_fsr_avg, mask_avg, mask_std = 0.0, 0.0, 0.0, 0.0, 0.0
        sem_plus, sem_minus_ade, sem_rec, sem_star, sem_gap = 0.0, 0.0, 0.0, 0.0, 0.0
        count = 0
        for i, data in enumerate(trainer.train_loader):
            clean = clone_batch(data)
            noisy = corrupt_batch(
                clean,
                noise_sigma=ntr,
                smooth_deg=-1,
                seed=seed,
                batch_idx=epoch * loader_len + i,
            )
            proc_c = trainer.data_preprocess(clean)
            proc_n = trainer.data_preprocess(noisy)
            fut = proc_n["fut_traj"]

            trainer.optimizer.zero_grad()
            raw_out = trainer.mrf_predictor(proc_n, trainer.N, vae_train=False, noise=True)
            if isinstance(raw_out, tuple):
                pred, dist_samples = raw_out[0], raw_out[1]
            else:
                pred, dist_samples = raw_out, None
            loss_traj = _ade_loss(pred, fut, reweight)

            z1 = None
            if (
                getattr(config, "fsr_sem", False)
                and dist_samples is not None
                and len(dist_samples) > 0
            ):
                z1 = (dist_samples[0] * trainer.mrf_predictor.sigma1).reshape(
                    -1, trainer.mrf_predictor.z_dim
                ).detach()

            loss_fsr = torch.zeros((), device=fut.device)
            use_sem = bool(getattr(config, "use_fsr", False) and getattr(config, "fsr_sem", False))
            use_feat = bool(
                getattr(config, "use_fsr", False) and getattr(trainer, "teacher", None) is not None
            )
            if use_sem or use_feat:
                loss_fsr = compute_fsr_loss(
                    trainer.mrf_predictor,
                    trainer.teacher,
                    proc_c,
                    proc_n,
                    trainer.traj_scale,
                    trainer.N,
                    config,
                    reweight=reweight,
                    z1=z1,
                    fut=fut,
                )
            loss = loss_traj + loss_fsr
            loss.backward()
            trainer.optimizer.step()

            loss_total += loss.item()
            loss_rcn += loss_traj.item()
            loss_fsr_avg += float(loss_fsr.item())
            aux = getattr(trainer.mrf_predictor, "_fsr_aux", None)
            if aux is not None and "mask" in aux:
                mask_avg += float(aux["mask"].mean().item())
                mask_std += float(aux["mask"].std().item())
            stats = getattr(trainer.mrf_predictor, "_fsr_sem_stats", None)
            if stats is not None:
                sem_plus += stats["plus"]
                sem_minus_ade += stats["minus_ade"]
                sem_rec += stats["rec"]
                sem_star += stats.get("star", 0.0)
                sem_gap += stats.get("gap", 0.0)
            count += 1

        log_msg = "[{}] Epoch: {}\tLoss: {:.6f}\tLoss Traj.: {:.6f}\tFSR: {:.6f}\tmask: {:.3f}\tmstd: {:.3f}".format(
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            epoch,
            loss_total / count,
            loss_rcn / count,
            loss_fsr_avg / count,
            mask_avg / count,
            mask_std / count,
        )
        if getattr(config, "fsr_sem", False):
            log_msg += "\tZ+: {:.4f}\tZ-: {:.4f}\tZ*: {:.4f}\tgap: {:.4f}\trec: {:.4f}".format(
                sem_plus / count,
                sem_minus_ade / count,
                sem_star / count,
                sem_gap / count,
                sem_rec / count,
            )
        trainer.print_log(log_msg, trainer.log)

        performance, samples = eval_plugin(trainer)
        _log_eval(trainer, performance, samples)
        score = _select_score(trainer, performance, samples)
        if score < best:
            best = score
            trainer.print_log("--best select ADE: {:.4f}".format(best), trainer.log)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": trainer.mrf_predictor.state_dict(),
                    "ADE": best,
                },
                best_path,
            )

        last_path = os.path.join(save_dir, "{}_ckpt_{}.pth".format(config.dataset, epoch))
        torch.save(
            {"epoch": epoch, "state_dict": trainer.mrf_predictor.state_dict()},
            last_path,
        )
        if epoch > 0:
            prev = os.path.join(save_dir, "{}_ckpt_{}.pth".format(config.dataset, epoch - 1))
            if os.path.isfile(prev):
                os.remove(prev)

        trainer.scheduler_model.step()


def test_plugin(trainer):
    config = trainer.config
    ckpt = getattr(config, "ckpt_path", "") or official_ckpt(
        config.dataset, config.log or "default"
    )
    if not os.path.isfile(ckpt):
        raise FileNotFoundError("Test checkpoint not found: {}".format(ckpt))
    load_state_dict_flexible(
        trainer.mrf_predictor, ckpt, strict=not bool(getattr(config, "use_fsr", False))
    )
    trainer.mrf_predictor.eval()
    trainer.print_log("[INFO] Loading model from: {}".format(ckpt), trainer.log)

    sigma = float(getattr(config, "eval_noise_sigma", 0.0) or 0.0)
    sm = int(getattr(config, "smooth_past_deg", -1))
    proto = str(getattr(config, "noise_protocol", "gaussian") or "gaussian")
    performance, samples = eval_plugin(trainer, noise_sigma=sigma, smooth_deg=sm)
    prefix = "[proto={} noise={:g} smooth={}] ".format(proto, sigma, sm)
    _log_eval(trainer, performance, samples, prefix=prefix)
    return performance, samples


def maybe_plugin_test(trainer) -> bool:
    if not should_use_plugin_test(trainer.config):
        return False
    test_plugin(trainer)
    return True
