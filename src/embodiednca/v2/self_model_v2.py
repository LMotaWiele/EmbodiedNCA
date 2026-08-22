"""Same SelfModel architecture as v1, retrained on combined damage+boost. Check 1 is boost-subset only."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from embodiednca.v2.boost_rollout_dataset import (
    PERT_BOOST,
    PERT_TYPE_NAMES,
    RolloutTensorDataset,
    filter_by_pert_type,
    train_val_split,
)

# Re-export the same architecture as v1 (identical class definition).
from embodiednca.v1.self_model import (
    SelfModel,
    mae,
    pearson_r,
    self_model_loss,
)


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_frac: float = 0.15
    seed: int = 0
    device: str = "cpu"
    hidden: int = 32


def train_self_model_v2(
    data: Dict[str, torch.Tensor],
    tcfg: Optional[TrainConfig] = None,
    channels: int = 32,
    verbose: bool = True,
) -> Tuple[SelfModel, Dict]:
    """Train on combined damage+boost data. Returns model + history + val_data."""
    tcfg = tcfg or TrainConfig()
    torch.manual_seed(tcfg.seed)

    train_data, val_data = train_val_split(data, val_frac=tcfg.val_frac, seed=tcfg.seed)
    train_loader = DataLoader(
        RolloutTensorDataset(train_data),
        batch_size=tcfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        RolloutTensorDataset(val_data),
        batch_size=tcfg.batch_size,
        shuffle=False,
    )

    model = SelfModel(channels=channels, hidden=tcfg.hidden).to(tcfg.device)
    opt = torch.optim.Adam(
        model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_mse_coh": [],
        "val_mse_lr": [],
        "val_bce_surv": [],
    }

    for epoch in range(1, tcfg.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            S = batch["S"].to(tcfg.device)
            R = batch["R"].to(tcfg.device)
            P = batch["pert_map"].to(tcfg.device)
            y_coh = batch["coherence_future"].to(tcfg.device)
            y_lr = batch["local_R_future"].to(tcfg.device)
            y_surv = batch["survived"].to(tcfg.device)

            opt.zero_grad(set_to_none=True)
            p_coh, p_lr, p_surv = model(S, R, P)
            loss, stats = self_model_loss(p_coh, p_lr, p_surv, y_coh, y_lr, y_surv)
            loss.backward()
            opt.step()
            train_losses.append(stats["loss"])

        model.eval()
        val_stats = {"loss": [], "mse_coh": [], "mse_lr": [], "bce_surv": []}
        with torch.no_grad():
            for batch in val_loader:
                S = batch["S"].to(tcfg.device)
                R = batch["R"].to(tcfg.device)
                P = batch["pert_map"].to(tcfg.device)
                y_coh = batch["coherence_future"].to(tcfg.device)
                y_lr = batch["local_R_future"].to(tcfg.device)
                y_surv = batch["survived"].to(tcfg.device)
                p_coh, p_lr, p_surv = model(S, R, P)
                _, stats = self_model_loss(p_coh, p_lr, p_surv, y_coh, y_lr, y_surv)
                val_stats["loss"].append(stats["loss"])
                val_stats["mse_coh"].append(stats["mse_coh"])
                val_stats["mse_lr"].append(stats["mse_lr"])
                val_stats["bce_surv"].append(stats["bce_surv"])

        def mean(xs: List[float]) -> float:
            return float(sum(xs) / max(1, len(xs)))

        tr = mean(train_losses)
        vl = mean(val_stats["loss"])
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["val_mse_coh"].append(mean(val_stats["mse_coh"]))
        history["val_mse_lr"].append(mean(val_stats["mse_lr"]))
        history["val_bce_surv"].append(mean(val_stats["bce_surv"]))
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == tcfg.epochs):
            print(
                f"  epoch {epoch:3d}/{tcfg.epochs}  "
                f"train={tr:.4f}  val={vl:.4f}  "
                f"mse_coh={history['val_mse_coh'][-1]:.4f}  "
                f"mse_lr={history['val_mse_lr'][-1]:.4f}  "
                f"bce={history['val_bce_surv'][-1]:.4f}"
            )

    return model, {
        "history": history,
        "n_train": train_data["S"].shape[0],
        "n_val": val_data["S"].shape[0],
        "train_idx_seed": tcfg.seed,
        "val_data": val_data,
    }


@torch.no_grad()
def evaluate_calibration(
    model: SelfModel,
    data: Dict[str, torch.Tensor],
    device: str,
    batch_size: int = 128,
    label: str = "all",
) -> Dict:
    """
    Held-out calibration: Pearson r / MAE for coherence and local_R.

    Same methodology as v1, plus distance-bin breakout. Call separately on
    the boost subset — do not pool with damage for the v2 pass bar.
    """
    model.eval()
    if data["S"].shape[0] == 0:
        return {
            "label": label,
            "n": 0,
            "coherence_r_all": float("nan"),
            "coherence_mae_all": float("nan"),
            "local_R_r_all": float("nan"),
            "local_R_mae_all": float("nan"),
            "by_distance": {},
            "pass_within_bin": False,
        }

    preds_coh, preds_lr, preds_surv = [], [], []
    trues_coh, trues_lr, trues_surv = [], [], []

    n = data["S"].shape[0]
    for i0 in range(0, n, batch_size):
        i1 = min(n, i0 + batch_size)
        S = data["S"][i0:i1].to(device)
        R = data["R"][i0:i1].to(device)
        P = data["pert_map"][i0:i1].to(device)
        p_coh, p_lr, p_surv = model(S, R, P)
        preds_coh.append(p_coh.cpu())
        preds_lr.append(p_lr.cpu())
        preds_surv.append(p_surv.cpu())
        trues_coh.append(data["coherence_future"][i0:i1])
        trues_lr.append(data["local_R_future"][i0:i1])
        trues_surv.append(data["survived"][i0:i1])

    pred_coh = torch.cat(preds_coh)
    true_coh = torch.cat(trues_coh)
    pred_lr = torch.cat(preds_lr)
    true_lr = torch.cat(trues_lr)
    pred_surv = torch.cat(preds_surv)
    true_surv = torch.cat(trues_surv)

    out: Dict = {
        "label": label,
        "n": n,
        "coherence_r_all": pearson_r(pred_coh, true_coh),
        "coherence_mae_all": mae(pred_coh, true_coh),
        "local_R_r_all": pearson_r(pred_lr, true_lr),
        "local_R_mae_all": mae(pred_lr, true_lr),
        "survival_acc": float(
            ((pred_surv >= 0.5).float() == true_surv).float().mean().item()
        ),
        "pred_coherence": pred_coh,
        "true_coherence": true_coh,
        "pred_local_R": pred_lr,
        "true_local_R": true_lr,
    }

    # Distance bins (same edges as v1)
    by_dist = {}
    within_rs = []
    if "distance_to_well" in data:
        dist = data["distance_to_well"]
        bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13)]
        for lo, hi in bins:
            m = (dist >= lo) & (dist < hi)
            bin_label = f"[{lo},{hi})"
            if int(m.sum().item()) < 5:
                by_dist[bin_label] = {
                    "n": int(m.sum().item()),
                    "coherence_r": None,
                    "coherence_mae": None,
                    "local_R_r": None,
                    "local_R_mae": None,
                }
                continue
            cr = pearson_r(pred_coh[m], true_coh[m])
            by_dist[bin_label] = {
                "n": int(m.sum().item()),
                "coherence_r": cr,
                "coherence_mae": mae(pred_coh[m], true_coh[m]),
                "local_R_r": pearson_r(pred_lr[m], true_lr[m]),
                "local_R_mae": mae(pred_lr[m], true_lr[m]),
                "true_coh_mean": float(true_coh[m].mean().item()),
                "true_coh_std": float(true_coh[m].std().item()),
                "true_local_R_mean": float(true_lr[m].mean().item()),
                "true_local_R_std": float(true_lr[m].std().item()),
            }
            if cr is not None and not math.isnan(cr):
                within_rs.append(cr)
    out["by_distance"] = by_dist
    out["coherence_r_within_bin_mean"] = (
        float(sum(within_rs) / len(within_rs)) if within_rs else float("nan")
    )

    # Pass bar for boost: mean within-bin r roughly ≥ 0.5 (hard fail <0.5);
    # target is comparable to v1 damage (~0.7+).
    mean_r = out["coherence_r_within_bin_mean"]
    out["pass_within_bin"] = (not math.isnan(mean_r)) and mean_r >= 0.5
    out["strong_pass_within_bin"] = (not math.isnan(mean_r)) and mean_r >= 0.7

    return out


@torch.no_grad()
def evaluate_boost_calibration(
    model: SelfModel,
    val_data: Dict[str, torch.Tensor],
    device: str,
) -> Dict:
    """
    Mandatory gate for v2: boost-subset calibration only (not pooled with damage).
    Also reports damage-subset and all for comparison.
    """
    boost = filter_by_pert_type(val_data, PERT_BOOST)
    cal_boost = evaluate_calibration(model, boost, device=device, label="boost")

    # Damage = everything that is not boost (null + noise + zero)
    if "pert_type" in val_data:
        m = val_data["pert_type"] != PERT_BOOST
        damage = {k: v[m] for k, v in val_data.items() if isinstance(v, torch.Tensor)}
        cal_damage = evaluate_calibration(model, damage, device=device, label="damage")
    else:
        cal_damage = {"n": 0, "label": "damage"}

    cal_all = evaluate_calibration(model, val_data, device=device, label="all")

    # Per-type breakdown
    by_type = {}
    if "pert_type" in val_data:
        for code, name in PERT_TYPE_NAMES.items():
            sub = filter_by_pert_type(val_data, code)
            if sub["S"].shape[0] >= 5:
                by_type[name] = evaluate_calibration(
                    model, sub, device=device, label=name
                )

    return {
        "boost": cal_boost,
        "damage": cal_damage,
        "all": cal_all,
        "by_type": {
            k: {
                "n": v["n"],
                "coherence_r_all": v["coherence_r_all"],
                "coherence_mae_all": v["coherence_mae_all"],
                "coherence_r_within_bin_mean": v.get("coherence_r_within_bin_mean"),
                "local_R_r_all": v["local_R_r_all"],
                "local_R_mae_all": v["local_R_mae_all"],
            }
            for k, v in by_type.items()
        },
    }


def save_model(model: SelfModel, path: Path | str, extra: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "channels": model.channels,
        "hidden": model.hidden,
        "version": "v2",
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_model(path: Path | str, device: str = "cpu") -> SelfModel:
    data = torch.load(path, map_location=device, weights_only=False)
    model = SelfModel(channels=data["channels"], hidden=data["hidden"]).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_coherence_future(
    model: SelfModel,
    S: torch.Tensor,
    R: torch.Tensor,
    pert_map: torch.Tensor,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Predicted future coherence for one or more (S, R, boost encoding) queries.

    Returns [B] tensor.
    """
    model.eval()
    device = device or next(model.parameters()).device
    if S.dim() == 3:
        S = S.unsqueeze(0)
    if R.dim() == 2:
        R = R.view(1, 1, *R.shape)
    elif R.dim() == 3:
        R = R.unsqueeze(0)
    if pert_map.dim() == 3:
        pert_map = pert_map.unsqueeze(0)
    p_coh, _, _ = model(S.to(device), R.to(device), pert_map.to(device))
    return p_coh
