"""Read-only predictor: (S, R, pert) → future coherence, local R, survival. Does not write the world."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from embodiednca.v1.rollout_dataset import RolloutTensorDataset, train_val_split


class SelfModel(nn.Module):
    """Compact conv encoder over [S | R | pert_map] → 3 heads. Smaller than SimpleNCA on purpose."""

    def __init__(self, channels: int = 32, hidden: int = 32):
        super().__init__()
        in_ch = channels + 1 + 1  # S + R + perturbation map
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, stride=2),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        self.channels = channels
        self.hidden = hidden

    def forward(
        self,
        S: torch.Tensor,
        R: torch.Tensor,
        perturbation_encoding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            S: [B, C, H, W]
            R: [B, 1, H, W]
            perturbation_encoding: [B, 1, H, W]  (Gaussian bump or zeros)

        Returns:
            pred_coherence: [B] in roughly [-1, 1] via tanh
            pred_local_R: [B] in [0, 1] via sigmoid
            pred_survival_prob: [B] in (0, 1) via sigmoid
        """
        x = torch.cat([S, R, perturbation_encoding], dim=1)
        h = self.enc(x)
        h = h.mean(dim=(-2, -1))  # global average pool
        out = self.head(h)
        pred_coh = torch.tanh(out[:, 0])
        pred_lr = torch.sigmoid(out[:, 1])
        pred_surv = torch.sigmoid(out[:, 2])
        return pred_coh, pred_lr, pred_surv


def self_model_loss(
    pred_coh: torch.Tensor,
    pred_lr: torch.Tensor,
    pred_surv: torch.Tensor,
    true_coh: torch.Tensor,
    true_lr: torch.Tensor,
    true_surv: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_coh = F.mse_loss(pred_coh, true_coh)
    loss_lr = F.mse_loss(pred_lr, true_lr)
    loss_surv = F.binary_cross_entropy(pred_surv, true_surv)
    loss = loss_coh + loss_lr + loss_surv
    return loss, {
        "loss": float(loss.item()),
        "mse_coh": float(loss_coh.item()),
        "mse_lr": float(loss_lr.item()),
        "bce_surv": float(loss_surv.item()),
    }


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


def train_self_model(
    data: Dict[str, torch.Tensor],
    tcfg: Optional[TrainConfig] = None,
    channels: int = 32,
    verbose: bool = True,
) -> Tuple[SelfModel, Dict]:
    """Train/val split + supervised training. Returns model + history dict."""
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
        "val_data": val_data,  # held-out for check 1
    }


@torch.no_grad()
def predict_batch(
    model: SelfModel,
    S: torch.Tensor,
    R: torch.Tensor,
    P: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    return model(S.to(device), R.to(device), P.to(device))


def pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation; returns nan if variance too small."""
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    if x.numel() < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom.item() < 1e-12:
        return float("nan")
    return float((x @ y / denom).item())


def mae(x: torch.Tensor, y: torch.Tensor) -> float:
    return float((x - y).abs().mean().item())


@torch.no_grad()
def evaluate_calibration(
    model: SelfModel,
    data: Dict[str, torch.Tensor],
    device: str,
    batch_size: int = 128,
) -> Dict:
    """
    Held-out calibration: predicted vs true coherence_future (and local_R, survival).

    Reports Pearson r / MAE on all examples (dead targets stored as 0) and on
    the subset where the ground-truth coherence was finite (not post-death).
    """
    model.eval()
    preds_coh, preds_lr, preds_surv = [], [], []
    trues_coh, trues_lr, trues_surv = [], [], []
    was_nan = []

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
        if "coherence_was_nan" in data:
            was_nan.append(data["coherence_was_nan"][i0:i1])

    pred_coh = torch.cat(preds_coh)
    true_coh = torch.cat(trues_coh)
    pred_lr = torch.cat(preds_lr)
    true_lr = torch.cat(trues_lr)
    pred_surv = torch.cat(preds_surv)
    true_surv = torch.cat(trues_surv)

    out = {
        "n": n,
        "coherence_r_all": pearson_r(pred_coh, true_coh),
        "coherence_mae_all": mae(pred_coh, true_coh),
        "local_R_r_all": pearson_r(pred_lr, true_lr),
        "local_R_mae_all": mae(pred_lr, true_lr),
        "survival_bce": float(
            F.binary_cross_entropy(pred_surv, true_surv).item()
        ),
        "survival_acc": float(((pred_surv >= 0.5).float() == true_surv).float().mean().item()),
        "pred_coherence": pred_coh,
        "true_coherence": true_coh,
        "pred_local_R": pred_lr,
        "true_local_R": true_lr,
        "pred_survival": pred_surv,
        "true_survival": true_surv,
    }

    if was_nan:
        nan_flag = torch.cat(was_nan) > 0.5
        alive = ~nan_flag
        if alive.any():
            out["coherence_r_alive"] = pearson_r(pred_coh[alive], true_coh[alive])
            out["coherence_mae_alive"] = mae(pred_coh[alive], true_coh[alive])
            out["n_alive"] = int(alive.sum().item())
        else:
            out["coherence_r_alive"] = float("nan")
            out["coherence_mae_alive"] = float("nan")
            out["n_alive"] = 0
    else:
        out["coherence_r_alive"] = out["coherence_r_all"]
        out["coherence_mae_alive"] = out["coherence_mae_all"]
        out["n_alive"] = n

    # --- Conditioned / within-bin diagnostics (honest check-1 stress tests) ---
    low_r = true_lr < 0.2
    high_r = true_lr >= 0.2
    out["local_R_mae_true_lt_0.2"] = (
        mae(pred_lr[low_r], true_lr[low_r]) if low_r.any() else float("nan")
    )
    out["local_R_mae_true_ge_0.2"] = (
        mae(pred_lr[high_r], true_lr[high_r]) if high_r.any() else float("nan")
    )
    out["local_R_r_true_lt_0.2"] = (
        pearson_r(pred_lr[low_r], true_lr[low_r]) if low_r.sum() >= 2 else float("nan")
    )
    out["n_local_R_lt_0.2"] = int(low_r.sum().item())
    if low_r.any():
        err_low = (pred_lr[low_r] - true_lr[low_r]).abs()
        out["local_R_p95_abs_err_true_lt_0.2"] = float(torch.quantile(err_low, 0.95).item())
        out["local_R_max_abs_err_true_lt_0.2"] = float(err_low.max().item())
        out["local_R_frac_pred_gt_0.2_when_true_lt_0.05"] = float(
            ((true_lr < 0.05) & (pred_lr > 0.2)).float().sum().item()
            / max(1, int((true_lr < 0.05).sum().item()))
        )
    else:
        out["local_R_p95_abs_err_true_lt_0.2"] = float("nan")
        out["local_R_max_abs_err_true_lt_0.2"] = float("nan")
        out["local_R_frac_pred_gt_0.2_when_true_lt_0.05"] = float("nan")

    # Distance bins (if present on the dataset dict)
    if "distance_to_well" in data:
        dist = data["distance_to_well"]
        bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 13)]
        by_dist = {}
        for lo, hi in bins:
            m = (dist >= lo) & (dist < hi)
            label = f"[{lo},{hi})"
            if m.sum() < 5:
                by_dist[label] = {"n": int(m.sum().item()), "coherence_r": None, "coherence_mae": None}
                continue
            by_dist[label] = {
                "n": int(m.sum().item()),
                "coherence_r": pearson_r(pred_coh[m], true_coh[m]),
                "coherence_mae": mae(pred_coh[m], true_coh[m]),
                "local_R_r": pearson_r(pred_lr[m], true_lr[m]),
                "local_R_mae": mae(pred_lr[m], true_lr[m]),
                "true_coh_mean": float(true_coh[m].mean().item()),
                "true_coh_std": float(true_coh[m].std().item()),
                "true_local_R_mean": float(true_lr[m].mean().item()),
                "true_local_R_std": float(true_lr[m].std().item()),
            }
        out["by_distance"] = by_dist

    # Perturbation type: null (mag=0) vs nonzero
    if "pert_mag" in data:
        mag = data["pert_mag"]
        by_pert = {}
        for label, m in (("null", mag <= 0), ("nonzero", mag > 0)):
            if m.sum() < 5:
                by_pert[label] = {"n": int(m.sum().item())}
                continue
            by_pert[label] = {
                "n": int(m.sum().item()),
                "coherence_r": pearson_r(pred_coh[m], true_coh[m]),
                "coherence_mae": mae(pred_coh[m], true_coh[m]),
                "local_R_r": pearson_r(pred_lr[m], true_lr[m]),
                "local_R_mae": mae(pred_lr[m], true_lr[m]),
            }
        out["by_perturbation"] = by_pert

    return out


def save_model(model: SelfModel, path: Path | str, extra: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "channels": model.channels,
        "hidden": model.hidden,
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
