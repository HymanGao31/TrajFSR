"""TrajFSR plugin helpers for NMRF """

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

from models.model_mrf_stride_sample import MRF_CVAE


DEFAULT_RESULTS_ROOT = "/root/autodl-tmp/nmrfFSR"

OFFICIAL_EPOCH = {
    "eth": 47,
    "hotel": 44,
    "univ": 15,
    "zara1": 47,
    "zara2": 55,
    "nba": 14,
    "sdd": 80,
    "jrdb": 49,
}


def horizon_cfg(dataset):
    """Official multi-horizon ADE/FDE. ETH/UCY/SDD stay single ADE@4.8s."""
    if dataset == "nba":
        return {"step": 5, "secs": (1.0, 2.0, 3.0, 4.0)}
    if dataset == "jrdb":
        return {"step": 3, "secs": (1.2, 2.4, 3.6, 4.8)}
    return None


def official_ckpt(dataset: str, log: str = "default") -> str:
    epoch = OFFICIAL_EPOCH[dataset]
    return os.path.join("results", dataset, log, "model_{:04d}.p".format(epoch))


def is_plugin_finetune(config) -> bool:
    return bool(getattr(config, "train", False)) and (
        bool(getattr(config, "use_fsr", False))
        or bool(getattr(config, "fsr", False))
        or float(getattr(config, "train_noise_sigma", 0) or 0) > 0
        or bool(getattr(config, "load_full", False))
    )


NOISE_PROTOCOLS = ("gaussian", "poisson", "mixed", "rand_sigma", "clean")


def should_use_plugin_test(config) -> bool:
    proto = str(getattr(config, "noise_protocol", "gaussian") or "gaussian").lower()
    return bool(
        getattr(config, "ckpt_path", "")
        or getattr(config, "use_fsr", False)
        or getattr(config, "fsr", False)
        or float(getattr(config, "eval_noise_sigma", 0) or 0) > 0
        or int(getattr(config, "smooth_past_deg", -1)) >= 0
        or proto not in ("gaussian", "", "clean")
    )


def add_fsr_cli_args(parser):
    parser.add_argument("--fsr", action="store_true", default=False)
    parser.add_argument("--freeze_backbone", action="store_true", default=False)
    parser.add_argument("--freeze_encoder", action="store_true", default=False,
                        help="Freeze History Encoder only; train FSR + decoder/sampler.")
    parser.add_argument("--load_full", action="store_true", default=False)
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--teacher_ckpt", type=str, default="")
    parser.add_argument("--exp", type=str, default="")
    parser.add_argument("--results_root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--num_epochs", type=int, default=-1)
    parser.add_argument("--fsr_tau", type=float, default=0.1)
    parser.add_argument("--lam_align", type=float, default=1.0)
    parser.add_argument("--lam_sep", type=float, default=1.0)
    parser.add_argument("--lam_rec", type=float, default=1.0)
    parser.add_argument("--lam_id", type=float, default=0.0,
                        help="Unused. Identity loss is not part of the method.")
    parser.add_argument("--lam_mask_bal", type=float, default=0.0,
                        help="Penalize noisy-mask mean away from 0.5 to stop M→1 collapse. 0 disables.")
    parser.add_argument("--soft_tau", type=float, default=1.0)
    parser.add_argument("--fsr_loss_scale", type=float, default=1.0)
    parser.add_argument("--fsr_sem", action="store_true", default=False,
                        help="Decoder semantic FSR: Z+ minADE, Z- two-sided ADE band, Z* must beat Z+.")
    parser.add_argument("--fsr_sem_m_lo", type=float, default=0.05,
                        help="Z- must be at least this much worse than Z+ (normalized ADE).")
    parser.add_argument("--fsr_sem_m_hi", type=float, default=0.20,
                        help="Z- must be at most this much worse than Z+ (normalized ADE).")
    parser.add_argument("--fsr_sem_delta", type=float, default=0.02,
                        help="Z*=Z++rec must beat Z+ by this much (normalized ADE).")
    parser.add_argument("--train_noise_sigma", type=float, default=0.0)
    parser.add_argument("--eval_noise_sigma", type=float, default=0.0)
    parser.add_argument("--smooth_past_deg", type=int, default=-1)
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument(
        "--noise_protocol",
        type=str,
        default="gaussian",
        help="Eval obs noise: gaussian (default, uses --eval_noise_sigma) | poisson | mixed | rand_sigma | clean.",
    )
    parser.add_argument("--poisson_lam", type=float, default=0.4)
    parser.add_argument("--mixed_gaussian_sigma", type=float, default=0.2)
    parser.add_argument("--mixed_poisson_lam", type=float, default=0.2)
    parser.add_argument("--gpu", type=str, default="")
    return parser


def apply_fsr_config(config):
    config.use_fsr = bool(getattr(config, "fsr", False))
    config.fsr_sem = bool(getattr(config, "fsr_sem", False))
    plugin = is_plugin_finetune(config) or bool(config.use_fsr) or bool(getattr(config, "load_full", False))
    if plugin and not getattr(config, "ckpt_path", ""):
        config.ckpt_path = official_ckpt(config.dataset, config.log or "default")
    if plugin and not getattr(config, "teacher_ckpt", ""):
        config.teacher_ckpt = config.ckpt_path or official_ckpt(config.dataset, config.log or "default")
    if is_plugin_finetune(config) and not getattr(config, "log", ""):
        config.log = getattr(config, "exp", "") or "nmrf_plugin"
    if getattr(config, "num_epochs", -1) is None:
        config.num_epochs = -1
    proto = str(getattr(config, "noise_protocol", "gaussian") or "gaussian").strip().lower()
    if proto in ("", "none", "default"):
        proto = "gaussian"
    if proto not in NOISE_PROTOCOLS:
        raise ValueError("noise_protocol must be one of {}; got {}".format(NOISE_PROTOCOLS, proto))
    config.noise_protocol = proto
    return config


def clone_batch(data: dict) -> dict:
    out = {}
    for k, v in data.items():
        out[k] = v.clone() if torch.is_tensor(v) else v
    return out


def add_obs_noise(past: torch.Tensor, sigma: float, seed: Optional[int] = None, batch_idx: int = 0) -> torch.Tensor:
    if sigma is None or float(sigma) <= 0:
        return past
    out = past.clone()
    if seed is not None:
        generator = torch.Generator(device=out.device)
        generator.manual_seed(int(seed) + int(batch_idx) * 1009)
        noise = torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype)
    else:
        noise = torch.randn_like(out)
    return out + noise * float(sigma)


def _poisson_centered(shape, lam, dtype, device, seed, batch_idx):
    """x' = x + (Poisson(lam) - lam), NATRA-style additive Poisson (meters on SDD)."""
    lam = float(lam)
    if seed is not None:
        torch.manual_seed(int(seed) + int(batch_idx) * 1009 + 7919)
    rate = torch.full(shape, lam, device=device, dtype=torch.float32)
    return torch.poisson(rate).to(dtype) - lam


def apply_natra_noise(
    past: torch.Tensor,
    protocol: str,
    seed: Optional[int] = None,
    batch_idx: int = 0,
    poisson_lam: float = 0.4,
    mixed_gaussian_sigma: float = 0.2,
    mixed_poisson_lam: float = 0.2,
    rand_sigma_choices=(0.2, 0.4),
) -> torch.Tensor:
    protocol = (protocol or "gaussian").lower()
    if protocol in ("gaussian", "", "clean"):
        return past
    out = past.clone()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=out.device)
        generator.manual_seed(int(seed) + int(batch_idx) * 1009)

    def _randn():
        if generator is None:
            return torch.randn_like(out)
        return torch.randn(out.shape, generator=generator, device=out.device, dtype=out.dtype)

    if protocol == "poisson":
        lam = float(poisson_lam)
        if lam <= 0:
            return out
        return out + _poisson_centered(out.shape, lam, out.dtype, out.device, seed, batch_idx)
    if protocol == "mixed":
        g_sigma = float(mixed_gaussian_sigma)
        p_lam = float(mixed_poisson_lam)
        if g_sigma > 0:
            out = out + _randn() * g_sigma
        if p_lam > 0:
            out = out + _poisson_centered(out.shape, p_lam, out.dtype, out.device, seed, batch_idx)
        return out
    if protocol == "rand_sigma":
        choices = tuple(float(s) for s in rand_sigma_choices)
        if not choices:
            return out
        n_ped = int(out.shape[0])
        idx = torch.randint(0, len(choices), (n_ped,), device=out.device, generator=generator)
        sigma_map = out.new_tensor(choices)[idx]
        view_shape = [n_ped] + [1] * (out.ndim - 1)
        return out + _randn() * sigma_map.view(*view_shape)
    raise ValueError("Unknown noise protocol: {!r}. Choose from {}.".format(protocol, NOISE_PROTOCOLS))


_SMOOTHER_CACHE = {}


def _poly_smoother(n, degree, device, dtype):
    key = (n, degree, str(device), str(dtype))
    smoother = _SMOOTHER_CACHE.get(key)
    if smoother is None:
        t = torch.linspace(-1.0, 1.0, n, device=device, dtype=torch.float64)
        basis = torch.stack([t ** k for k in range(degree + 1)], dim=1)
        smoother = (basis @ torch.linalg.pinv(basis)).to(dtype)
        _SMOOTHER_CACHE[key] = smoother
    return smoother


def smooth_past(past: torch.Tensor, degree: int) -> torch.Tensor:
    """Polynomial smooth over the time axis (dim=-2). past: [..., T, 2]."""
    if degree is None or int(degree) < 0:
        return past
    degree = int(degree)
    n = past.shape[-2]
    if degree + 1 >= n:
        return past
    smoother = _poly_smoother(n, degree, past.device, past.dtype)
    return torch.einsum("pq,...qc->...pc", smoother, past)


def corrupt_past(
    past: torch.Tensor,
    noise_sigma: float = 0.0,
    smooth_deg: int = -1,
    seed=None,
    batch_idx: int = 0,
    protocol: str = "gaussian",
    poisson_lam: float = 0.4,
    mixed_gaussian_sigma: float = 0.2,
    mixed_poisson_lam: float = 0.2,
    rand_sigma_choices=(0.2, 0.4),
):
    protocol = (protocol or "gaussian").lower()
    if protocol in ("gaussian", ""):
        x = add_obs_noise(past, noise_sigma, seed=seed, batch_idx=batch_idx)
    else:
        x = apply_natra_noise(
            past,
            protocol,
            seed=seed,
            batch_idx=batch_idx,
            poisson_lam=poisson_lam,
            mixed_gaussian_sigma=mixed_gaussian_sigma,
            mixed_poisson_lam=mixed_poisson_lam,
            rand_sigma_choices=rand_sigma_choices,
        )
    return smooth_past(x, smooth_deg)


def corrupt_batch(
    data: dict,
    noise_sigma: float = 0.0,
    smooth_deg: int = -1,
    seed=None,
    batch_idx: int = 0,
    protocol: str = "gaussian",
    poisson_lam: float = 0.4,
    mixed_gaussian_sigma: float = 0.2,
    mixed_poisson_lam: float = 0.2,
    rand_sigma_choices=(0.2, 0.4),
) -> dict:
    out = clone_batch(data)
    out["pre_motion_3D"] = corrupt_past(
        out["pre_motion_3D"],
        noise_sigma,
        smooth_deg,
        seed=seed,
        batch_idx=batch_idx,
        protocol=protocol,
        poisson_lam=poisson_lam,
        mixed_gaussian_sigma=mixed_gaussian_sigma,
        mixed_poisson_lam=mixed_poisson_lam,
        rand_sigma_choices=rand_sigma_choices,
    )
    return out


def unwrap_pred(out):
    if isinstance(out, tuple):
        return out[0]
    return out


def min_ade_mean(pred, fut, scale: float) -> torch.Tensor:
    """pred [K,P,T,2], fut [P,T,2] -> scalar mean minADE in original units."""
    dist = torch.norm(fut.unsqueeze(0) - pred, dim=-1) * scale
    return dist.mean(dim=-1).min(dim=0)[0].mean()


def load_state_dict_flexible(model: nn.Module, ckpt_path: str, strict: bool = True, drop_prefixes=()):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if drop_prefixes:
        prefixes = tuple(drop_prefixes)
        state = {k: v for k, v in state.items() if not str(k).startswith(prefixes)}
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    ret = model.load_state_dict(state, strict=strict)
    if ret is None:
        return epoch, [], []
    missing, unexpected = ret
    return epoch, list(missing), list(unexpected)


def build_save_dir(config, dataset: str) -> str:
    tag = getattr(config, "exp", "") or "nmrf"
    extras = []
    if getattr(config, "use_fsr", False):
        extras.append("FSR")
    if getattr(config, "fsr_sem", False):
        extras.append("sem")
        extras.append("band")
    if getattr(config, "freeze_encoder", False):
        extras.append("frzE")
    elif getattr(config, "freeze_backbone", False):
        extras.append("frz")
    lam_mb = float(getattr(config, "lam_mask_bal", 0.0) or 0.0)
    if abs(lam_mb) > 1e-8:
        extras.append("mb{:g}".format(lam_mb))
    fsc = float(getattr(config, "fsr_loss_scale", 1.0))
    if abs(fsc - 1.0) > 1e-8:
        extras.append("fsc{:g}".format(fsc))
    ntr = float(getattr(config, "train_noise_sigma", 0) or 0)
    if ntr > 0:
        extras.append("ntr{:g}".format(ntr))
    suffix = ("_" + "_".join(extras)) if extras else ""
    root = getattr(config, "results_root", DEFAULT_RESULTS_ROOT)
    save_dir = os.path.join(root, "{}{}{}".format(dataset, "_" + tag if tag else "", suffix))
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def apply_param_freeze(model: nn.Module, freeze_backbone: bool = False, freeze_encoder: bool = False):
    if freeze_encoder:
        for name, p in model.named_parameters():
            p.requires_grad = not name.startswith("encoder_past.")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print("[INFO] freeze encoder only | trainable/total = {}/{}".format(trainable, total))
        return
    if not freeze_backbone:
        return
    for name, p in model.named_parameters():
        p.requires_grad = "fsr" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("[INFO] freeze backbone (FSR only) | trainable/total = {}/{}".format(trainable, total))


def build_teacher(config, ckpt_path: str, fut_step: int, hyper: dict, n_samples: int) -> MRF_CVAE:
    teacher = MRF_CVAE(
        fut_step=fut_step,
        z_dim=hyper["z_dim"],
        f2_dim=hyper["f2_dim"],
        sigma1=1.0,
        sigma2=1.0,
        stride=hyper["stride"],
        N=n_samples,
        use_fsr=False,
    ).cuda()
    load_state_dict_flexible(teacher, ckpt_path, strict=False, drop_prefixes=("fsr.",))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _traj_minade(pred, fut, reweight) -> torch.Tensor:
    """Same minADE as plugin traj loss: pred [N,P,T,2], fut [P,T,2]."""
    return ((pred - fut.unsqueeze(0)).norm(p=2, dim=-1) * reweight).mean(dim=-1).min(dim=0)[0].mean()


def decode_from_history(model, proc, n_samples, hist_feat, z1=None):
    """Run sampler decoder on a history vector without re-applying FSR."""
    return unwrap_pred(
        model(
            proc,
            n_samples,
            vae_train=False,
            noise=True,
            history_feature_override=hist_feat,
            z1_override=z1,
        )
    )


def compute_fsr_sem_loss(student, noisy_dict, fut, reweight, z1, n_samples, config) -> torch.Tensor:
    """Semantic FSR without unbounded maxADE.

    Z+ minADE; Z- stays in [Z++m_lo, Z++m_hi] (log1p on the upper side so a
    destroyed Z- still has gradient); Z* = Z+ + rec must beat Z+ by delta.
    Decoding rec alone is only for logs — minADE(rec) would clone Z+.
    """
    aux = getattr(student, "_fsr_aux", None)
    device = noisy_dict["past_traj"].device
    if aux is None or student.fsr is None:
        return torch.zeros((), device=device)

    z_plus = aux["f+"].squeeze(1)
    z_minus = aux["f-"].squeeze(1)
    z_rec = aux["rec"].squeeze(1)
    z_star = z_plus + z_rec

    pred_p = decode_from_history(student, noisy_dict, n_samples, z_plus, z1)
    pred_m = decode_from_history(student, noisy_dict, n_samples, z_minus, z1)
    pred_r = decode_from_history(student, noisy_dict, n_samples, z_rec, z1)
    pred_s = decode_from_history(student, noisy_dict, n_samples, z_star, z1)

    ade_p = _traj_minade(pred_p, fut, reweight)
    ade_m = _traj_minade(pred_m, fut, reweight)
    ade_r = _traj_minade(pred_r, fut, reweight)
    ade_s = _traj_minade(pred_s, fut, reweight)

    m_lo = float(getattr(config, "fsr_sem_m_lo", 0.05))
    m_hi = float(getattr(config, "fsr_sem_m_hi", 0.20))
    delta = float(getattr(config, "fsr_sem_delta", 0.02))

    l_plus = ade_p
    l_minus = torch.relu(ade_p + m_lo - ade_m) + torch.log1p(torch.relu(ade_m - ade_p - m_hi))
    l_star = ade_s
    l_repair = torch.relu(ade_s - ade_p + delta)

    fsc = float(config.fsr_loss_scale)
    loss = fsc * (
        float(config.lam_align) * l_plus
        + float(config.lam_sep) * l_minus
        + float(config.lam_rec) * (l_star + l_repair)
    )
    student._fsr_sem_stats = {
        "plus": float(ade_p.detach()),
        "minus": float(l_minus.detach()),
        "minus_ade": float(ade_m.detach()),
        "rec": float(ade_r.detach()),
        "star": float(ade_s.detach()),
        "gap": float((ade_p - ade_s).detach()),
    }
    return loss


def compute_fsr_loss(
    student,
    teacher,
    clean_dict,
    noisy_dict,
    scale,
    n_samples,
    config,
    reweight=None,
    z1=None,
    fut=None,
) -> torch.Tensor:
    if bool(getattr(config, "fsr_sem", False)):
        return compute_fsr_sem_loss(
            student,
            noisy_dict,
            fut if fut is not None else noisy_dict["fut_traj"],
            reweight,
            z1,
            n_samples,
            config,
        )

    aux = getattr(student, "_fsr_aux", None)
    if aux is None or student.fsr is None:
        return torch.zeros((), device=clean_dict["past_traj"].device)

    lam_align = float(config.lam_align)
    lam_sep = float(config.lam_sep)
    lam_rec = float(config.lam_rec)
    lam_bal = float(getattr(config, "lam_mask_bal", 0.0) or 0.0)
    soft_tau = float(config.soft_tau)
    fsc = float(config.fsr_loss_scale)

    with torch.no_grad():
        z_t = teacher.encode_history(clean_dict).unsqueeze(1)
        pred_c = unwrap_pred(teacher(clean_dict, n_samples, vae_train=False, noise=True))
        pred_n = unwrap_pred(teacher(noisy_dict, n_samples, vae_train=False, noise=True))
        ade_c = min_ade_mean(pred_c, clean_dict["fut_traj"], scale)
        ade_n = min_ade_mean(pred_n, noisy_dict["fut_traj"], scale)
        s = torch.sigmoid(soft_tau - ade_n / (ade_c + 1e-6))

    residual = (aux["enc"] - z_t).detach()
    l_align = ((aux["f+"] - z_t) ** 2).mean()
    l_sep = ((aux["f-"] - residual) ** 2).mean()
    l_rec = ((aux["rec"] - z_t) ** 2).mean()
    # Noisy mask only: keep ~half the units in Z- so rec has something to repair.
    l_bal = (aux["mask"].mean() - 0.5) ** 2

    loss = fsc * (s * lam_align * l_align + (1.0 - s) * (lam_sep * l_sep + lam_rec * l_rec))
    loss = loss + lam_bal * l_bal
    return loss
