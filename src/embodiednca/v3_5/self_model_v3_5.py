"""SelfModel plus two wear channels so refill-vs-floored gen is distinguishable. New checkpoint, not an edit of v2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from embodiednca.v3_5.death_aware_rollout_dataset import (
    DeathAwareTensorDataset,
    PERT_BOOST,
    PERT_NULL,
    PERT_TYPE_NAMES,
    filter_by_pert_type,
    train_val_split,
)
from embodiednca.v1.self_model import mae, pearson_r


class SelfModelV35(nn.Module):
    """
    Compact conv encoder over [S | R | pert_map | wear_ch_0 | wear_ch_1] → 3 heads.

    Wear is broadcast as two constant spatial channels (one per well's current
    effective_multiplier by default). Same head structure as v1/v2.
    """

    def __init__(
        self,
        channels: int = 32,
        hidden: int = 32,
        n_wells: int = 2,
        wear_input: str = "multipliers",  # or "wear"
    ):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.n_wells = n_wells
        self.wear_input = wear_input
        in_ch = channels + 1 + 1 + n_wells  # S + R + pert + wear channels
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

    def _wear_channels(
        self,
        wear_or_mult: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        wear_or_mult: [B, n_wells] → [B, n_wells, H, W] constant maps.
        """
        if wear_or_mult.dim() == 1:
            wear_or_mult = wear_or_mult.unsqueeze(0)
        B, n = wear_or_mult.shape
        # Expand to spatial
        ch = wear_or_mult.view(B, n, 1, 1).expand(B, n, H, W)
        return ch

    def forward(
        self,
        S: torch.Tensor,
        R: torch.Tensor,
        perturbation_encoding: torch.Tensor,
        wear_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            S: [B, C, H, W]
            R: [B, 1, H, W]
            perturbation_encoding: [B, 1, H, W]
            wear_features: [B, n_wells]  (multipliers or raw wear)

        Returns:
            pred_coherence, pred_local_R, pred_survival_prob  — each [B]
        """
        if S.dim() == 3:
            S = S.unsqueeze(0)
        if R.dim() == 2:
            R = R.view(1, 1, *R.shape)
        elif R.dim() == 3:
            R = R.unsqueeze(0)
        if perturbation_encoding.dim() == 3:
            perturbation_encoding = perturbation_encoding.unsqueeze(0)
        if wear_features.dim() == 1:
            wear_features = wear_features.unsqueeze(0)

        B, _, H, W = S.shape
        wear_ch = self._wear_channels(wear_features, H, W)
        x = torch.cat([S, R, perturbation_encoding, wear_ch], dim=1)
        h = self.enc(x)
        h = h.mean(dim=(-2, -1))
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
    surv_pos_weight: Optional[float] = None,
    surv_loss_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    loss_coh = F.mse_loss(pred_coh, true_coh)
    loss_lr = F.mse_loss(pred_lr, true_lr)
    if surv_pos_weight is not None and surv_pos_weight > 0:
        # Weight death examples (survived=0) so minority class is not ignored.
        w = torch.ones_like(true_surv)
        death_w = float(surv_pos_weight)
        w = torch.where(true_surv < 0.5, torch.full_like(w, death_w), w)
        loss_surv = F.binary_cross_entropy(pred_surv, true_surv, weight=w)
    else:
        loss_surv = F.binary_cross_entropy(pred_surv, true_surv)
    loss = loss_coh + loss_lr + float(surv_loss_scale) * loss_surv
    return loss, {
        "loss": float(loss.item()),
        "mse_coh": float(loss_coh.item()),
        "mse_lr": float(loss_lr.item()),
        "bce_surv": float(loss_surv.item()),
    }


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_frac: float = 0.15
    seed: int = 0
    device: str = "cpu"
    hidden: int = 48  # slightly larger than v2 — extra wear inputs
    wear_input: str = "multipliers"  # "multipliers" or "wear"
    surv_loss_scale: float = 2.0  # upweight survival head vs coh/local_R


def _batch_wear_features(
    batch: Dict[str, torch.Tensor],
    wear_input: str,
    device: str,
) -> torch.Tensor:
    if wear_input == "wear":
        return batch["wear"].to(device)
    return batch["multipliers"].to(device)


def train_self_model_v3_5(
    data: Dict[str, torch.Tensor],
    tcfg: Optional[TrainConfig] = None,
    channels: int = 32,
    verbose: bool = True,
) -> Tuple[SelfModelV35, Dict]:
    tcfg = tcfg or TrainConfig()
    torch.manual_seed(tcfg.seed)

    train_data, val_data = train_val_split(data, val_frac=tcfg.val_frac, seed=tcfg.seed)

    # Oversample death examples in the *training* set so the survival head
    # sees a non-degenerate class mix each epoch (val left untouched).
    dead_m = train_data["survived"] < 0.5
    live_m = ~dead_m
    n_dead_tr = int(dead_m.sum().item())
    n_live_tr = int(live_m.sum().item())
    if n_dead_tr > 0 and n_live_tr > n_dead_tr:
        target_dead = min(n_live_tr, n_dead_tr * 4)  # up to 1:1 after 4×
        reps = max(1, int(math.ceil(target_dead / n_dead_tr)))
        dead_idx = torch.where(dead_m)[0]
        live_idx = torch.where(live_m)[0]
        dead_rep = dead_idx.repeat(reps)[:target_dead]
        g = torch.Generator().manual_seed(tcfg.seed + 7)
        live_perm = live_idx[torch.randperm(n_live_tr, generator=g)]
        # keep all live + oversampled dead
        new_idx = torch.cat([live_perm, dead_rep])
        new_idx = new_idx[torch.randperm(new_idx.numel(), generator=g)]
        train_data = {
            k: v[new_idx] for k, v in train_data.items() if isinstance(v, torch.Tensor)
        }
        if verbose:
            print(
                f"  oversampled deaths in train: {n_dead_tr} → "
                f"{int((train_data['survived'] < 0.5).sum().item())}  "
                f"(live={n_live_tr})"
            )

    train_loader = DataLoader(
        DeathAwareTensorDataset(train_data),
        batch_size=tcfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        DeathAwareTensorDataset(val_data),
        batch_size=tcfg.batch_size,
        shuffle=False,
    )

    n_wells = int(train_data["wear"].shape[1])
    model = SelfModelV35(
        channels=channels,
        hidden=tcfg.hidden,
        n_wells=n_wells,
        wear_input=tcfg.wear_input,
    ).to(tcfg.device)

    # Death-class weight for survival BCE (if imbalanced).
    # Cap moderately: extreme weights can destabilize but under-weighting
    # death (rare) collapses to always-survive.
    n_dead = float((train_data["survived"] < 0.5).sum().item())
    n_live = float((train_data["survived"] >= 0.5).sum().item())
    if n_dead > 0 and n_live > 0:
        death_weight = n_live / n_dead
        death_weight = float(min(max(death_weight, 2.0), 12.0))
    else:
        death_weight = None

    if verbose:
        print(
            f"  train n={train_data['S'].shape[0]}  val n={val_data['S'].shape[0]}  "
            f"n_wells={n_wells}  n_dead={int(n_dead)} n_live={int(n_live)}  "
            f"death_weight={death_weight}"
        )

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
            Wf = _batch_wear_features(batch, tcfg.wear_input, tcfg.device)
            y_coh = batch["coherence_future"].to(tcfg.device)
            y_lr = batch["local_R_future"].to(tcfg.device)
            y_surv = batch["survived"].to(tcfg.device)

            opt.zero_grad(set_to_none=True)
            p_coh, p_lr, p_surv = model(S, R, P, Wf)
            loss, stats = self_model_loss(
                p_coh,
                p_lr,
                p_surv,
                y_coh,
                y_lr,
                y_surv,
                surv_pos_weight=death_weight,
                surv_loss_scale=tcfg.surv_loss_scale,
            )
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
                Wf = _batch_wear_features(batch, tcfg.wear_input, tcfg.device)
                y_coh = batch["coherence_future"].to(tcfg.device)
                y_lr = batch["local_R_future"].to(tcfg.device)
                y_surv = batch["survived"].to(tcfg.device)
                p_coh, p_lr, p_surv = model(S, R, P, Wf)
                _, stats = self_model_loss(
                    p_coh,
                    p_lr,
                    p_surv,
                    y_coh,
                    y_lr,
                    y_surv,
                    surv_pos_weight=None,
                    surv_loss_scale=tcfg.surv_loss_scale,
                )
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
        "death_weight": death_weight,
        "wear_input": tcfg.wear_input,
    }


def balanced_accuracy(
    pred: torch.Tensor,
    true: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Balanced accuracy for binary survived labels (1=survived, 0=died)."""
    pred_bin = (pred >= threshold).float()
    # class 1 = survived, class 0 = died
    tp = ((pred_bin == 1) & (true == 1)).sum().item()
    tn = ((pred_bin == 0) & (true == 0)).sum().item()
    fp = ((pred_bin == 1) & (true == 0)).sum().item()
    fn = ((pred_bin == 0) & (true == 1)).sum().item()
    sens = tp / max(1.0, tp + fn)  # recall survived
    spec = tn / max(1.0, tn + fp)  # recall death
    return float(0.5 * (sens + spec))


def death_class_metrics(
    pred: torch.Tensor,
    true: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Metrics treating *death* as the positive class (died = 1 - survived).
    Precision/recall on death specifically — plain accuracy is not enough.
    """
    # Convert to death labels
    true_death = (true < 0.5).float()  # 1 if died
    pred_death = (pred < threshold).float()  # model predicts death if p_surv < thr

    tp = ((pred_death == 1) & (true_death == 1)).sum().item()
    fp = ((pred_death == 1) & (true_death == 0)).sum().item()
    fn = ((pred_death == 0) & (true_death == 1)).sum().item()
    tn = ((pred_death == 0) & (true_death == 0)).sum().item()

    prec = tp / max(1.0, tp + fp)
    rec = tp / max(1.0, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {
        "death_precision": float(prec),
        "death_recall": float(rec),
        "death_f1": float(f1),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "n_true_death": float(true_death.sum().item()),
        "n_true_survived": float((1.0 - true_death).sum().item()),
    }


@torch.no_grad()
def evaluate_calibration_v35(
    model: SelfModelV35,
    data: Dict[str, torch.Tensor],
    device: str,
    batch_size: int = 128,
    label: str = "all",
    wear_input: str = "multipliers",
) -> Dict:
    """
    Held-out calibration: coh/local_R Pearson+MAE plus survival balanced
    accuracy and death-class precision/recall.
    """
    model.eval()
    if data["S"].shape[0] == 0:
        return {
            "label": label,
            "n": 0,
            "coherence_r_all": float("nan"),
            "local_R_r_all": float("nan"),
            "survival_acc": float("nan"),
            "survival_balanced_acc": float("nan"),
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
        if wear_input == "wear":
            Wf = data["wear"][i0:i1].to(device)
        else:
            Wf = data["multipliers"][i0:i1].to(device)
        p_coh, p_lr, p_surv = model(S, R, P, Wf)
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

    death_m = death_class_metrics(pred_surv, true_surv)
    bal_acc = balanced_accuracy(pred_surv, true_surv)
    raw_acc = float(((pred_surv >= 0.5).float() == true_surv).float().mean().item())

    out: Dict = {
        "label": label,
        "n": n,
        "coherence_r_all": pearson_r(pred_coh, true_coh),
        "coherence_mae_all": mae(pred_coh, true_coh),
        "local_R_r_all": pearson_r(pred_lr, true_lr),
        "local_R_mae_all": mae(pred_lr, true_lr),
        "survival_acc": raw_acc,
        "survival_balanced_acc": bal_acc,
        "death_rate_true": float((true_surv < 0.5).float().mean().item()),
        "pred_coherence": pred_coh,
        "true_coherence": true_coh,
        "pred_local_R": pred_lr,
        "true_local_R": true_lr,
        "pred_survived": pred_surv,
        "true_survived": true_surv,
        **death_m,
    }

    by_dist = {}
    within_rs = []
    if "distance_to_well" in data:
        dist = data["distance_to_well"]
        bins = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 14)]
        for lo, hi in bins:
            m = (dist >= lo) & (dist < hi)
            bin_label = f"[{lo},{hi})"
            n_bin = int(m.sum().item())
            if n_bin < 5:
                by_dist[bin_label] = {"n": n_bin, "coherence_r": None}
                continue
            cr = pearson_r(pred_coh[m], true_coh[m])
            by_dist[bin_label] = {
                "n": n_bin,
                "coherence_r": cr,
                "coherence_mae": mae(pred_coh[m], true_coh[m]),
                "local_R_r": pearson_r(pred_lr[m], true_lr[m]),
                "local_R_mae": mae(pred_lr[m], true_lr[m]),
                "survival_balanced_acc": balanced_accuracy(pred_surv[m], true_surv[m]),
                "death_rate": float((true_surv[m] < 0.5).float().mean().item()),
            }
            if cr is not None and not math.isnan(cr):
                within_rs.append(cr)
    out["by_distance"] = by_dist
    out["coherence_r_within_bin_mean"] = (
        float(sum(within_rs) / len(within_rs)) if within_rs else float("nan")
    )
    mean_r = out["coherence_r_within_bin_mean"]
    out["pass_within_bin"] = (not math.isnan(mean_r)) and mean_r >= 0.5
    # Survival: meaningfully above chance (0.5); first honest baseline
    out["pass_survival"] = bal_acc > 0.5
    return out


def save_model(
    model: SelfModelV35,
    path: Path | str,
    extra: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "channels": model.channels,
        "hidden": model.hidden,
        "n_wells": model.n_wells,
        "wear_input": model.wear_input,
        "version": "v3.5",
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_model(path: Path | str, device: str = "cpu") -> SelfModelV35:
    data = torch.load(path, map_location=device, weights_only=False)
    model = SelfModelV35(
        channels=data["channels"],
        hidden=data["hidden"],
        n_wells=data.get("n_wells", 2),
        wear_input=data.get("wear_input", "multipliers"),
    ).to(device)
    model.load_state_dict(data["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_coherence_future(
    model: SelfModelV35,
    S: torch.Tensor,
    R: torch.Tensor,
    pert_map: torch.Tensor,
    wear_features: torch.Tensor,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Predicted future coherence; wear_features [B, n_wells] or [n_wells]."""
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
    if wear_features.dim() == 1:
        wear_features = wear_features.unsqueeze(0)
    p_coh, _, _ = model(
        S.to(device), R.to(device), pert_map.to(device), wear_features.to(device)
    )
    return p_coh
