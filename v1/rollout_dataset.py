"""
rollout_dataset.py
==================
Perturbation-rollout dataset for v1 self-model training.

Uses the v0 EmbodiedNCA simulator (ResourceGrid / trained SimpleNCA) **read-only**:
burn-in → record (S_t, R_t) → apply a local perturbation → roll H steps with no
further intervention → record future coherence / local R / death.

Does not modify grid_world step logic, SimpleNCA, or resource physics.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ---------------------------------------------------------------------------
# Import v0 package (sibling directory under EmbodiedNCA/)
# ---------------------------------------------------------------------------
V1_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V1_ROOT.parent  # EmbodiedNCA/
V0_ROOT = REPO_ROOT / "v0"
if str(V0_ROOT) not in sys.path:
    sys.path.insert(0, str(V0_ROOT))
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))

from grid_world import (  # noqa: E402
    DEFAULT_WELLS,
    ResourceGrid,
    WorldConfig,
    make_seed_pattern,
    place_pattern,
)
from metrics import (  # noqa: E402
    DEATH_MASS_EPS,
    DEFAULT_ACTIVATION_THRESHOLD,
    coherence,
    local_resource_at_pattern_centroid,
    self_mass,
)
from nca_module import SimpleNCA  # noqa: E402

DEFAULT_CKPT = V0_ROOT / "results" / "trained_nca.pt"
BASE_CENTER = (16, 16)


@dataclass
class DatasetConfig:
    n_examples: int = 12_000
    burn_in: int = 20
    horizon: int = 100
    dist_min: float = 0.0
    dist_max: float = 12.0
    null_frac: float = 0.15
    patch_sizes: Tuple[int, ...] = (3, 5)
    noise_std: float = 1.5
    seed: int = 0
    device: str = "cpu"
    wells: Tuple[Tuple[int, int], ...] = DEFAULT_WELLS


def load_v0_world(
    ckpt_path: Path | str = DEFAULT_CKPT,
    device: str = "cpu",
) -> Tuple[SimpleNCA, torch.Tensor, WorldConfig]:
    """Load trained SimpleNCA + base pattern; freeze NCA weights."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"v0 checkpoint not found: {ckpt_path}. "
            "Run: python v0/run_sanity_checks.py  (from EmbodiedNCA root)."
        )
    data = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = WorldConfig(device=device)
    nca = SimpleNCA(
        channels=cfg.channels,
        hidden_dim=cfg.hidden_dim,
        grid_size=cfg.H,
    ).to(device)
    nca.load_state_dict(data["nca"])
    nca.eval()
    for p in nca.parameters():
        p.requires_grad_(False)
    base_pattern = data["base_pattern"].to(device)
    return nca, base_pattern, cfg


def nearest_well_distance(y: float, x: float, wells: Sequence[Tuple[int, int]]) -> float:
    return min(math.hypot(y - wy, x - wx) for wy, wx in wells)


def sample_center_at_distance(
    wells: Sequence[Tuple[int, int]],
    H: int,
    W: int,
    dist: float,
    rng: torch.Generator,
    max_tries: int = 40,
) -> Tuple[int, int]:
    """
    Sample integer center near `dist` from a random well, clamped to the grid.
    Retries if the draw lands too close to another well or off-structure.
    """
    wells = list(wells)
    for _ in range(max_tries):
        wi = int(torch.randint(0, len(wells), (1,), generator=rng).item())
        wy, wx = wells[wi]
        angle = float(torch.rand(1, generator=rng).item() * 2.0 * math.pi)
        cy = int(round(wy + dist * math.sin(angle)))
        cx = int(round(wx + dist * math.cos(angle)))
        cy = max(2, min(H - 3, cy))
        cx = max(2, min(W - 3, cx))
        # Accept if nearest-well distance is roughly the requested dist
        d = nearest_well_distance(cy, cx, wells)
        if abs(d - dist) <= max(1.5, 0.25 * dist + 1.0):
            return cy, cx
    # Fallback: project along +x from first well
    wy, wx = wells[0]
    return max(2, min(H - 3, wy)), max(2, min(W - 3, wx + int(round(dist))))


def make_perturbation_map(
    H: int,
    W: int,
    y: int,
    x: int,
    magnitude: float,
    device: str | torch.device,
    sigma: float = 1.5,
) -> torch.Tensor:
    """
    Soft spatial encoding of a probe/perturbation: [1, 1, H, W] Gaussian bump
    peaked at (y, x) with amplitude = magnitude (0 ⇒ null).
    """
    if magnitude <= 0:
        return torch.zeros(1, 1, H, W, device=device)
    yy = torch.arange(H, device=device, dtype=torch.float32).view(H, 1)
    xx = torch.arange(W, device=device, dtype=torch.float32).view(1, W)
    dist2 = (yy - y) ** 2 + (xx - x) ** 2
    bump = torch.exp(-dist2 / (2.0 * sigma ** 2))
    return (magnitude * bump).view(1, 1, H, W)


def apply_perturbation(
    S: torch.Tensor,
    y: int,
    x: int,
    mode: str,
    k: int,
    noise_std: float,
    rng: torch.Generator,
) -> Tuple[torch.Tensor, float]:
    """
    Apply a local modification to S. Returns (S_perturbed, magnitude_scalar).

    mode:
      - "null": no change, magnitude 0
      - "noise": add N(0, noise_std) on k×k patch
      - "zero": zero the k×k patch
    """
    S2 = S.clone()
    if mode == "null":
        return S2, 0.0

    H, W = S2.shape[-2], S2.shape[-1]
    half = k // 2
    y0, y1 = max(0, y - half), min(H, y + half + 1)
    x0, x1 = max(0, x - half), min(W, x + half + 1)
    if mode == "zero":
        S2[:, :, y0:y1, x0:x1] = 0.0
        mag = 1.0
    elif mode == "noise":
        # Generator must live on CPU for randn with generator in some builds;
        # sample on CPU then move.
        shape = S2[:, :, y0:y1, x0:x1].shape
        noise = torch.randn(shape, generator=rng) * noise_std
        S2[:, :, y0:y1, x0:x1] = S2[:, :, y0:y1, x0:x1] + noise.to(S2.device)
        mag = float(noise_std)
    else:
        raise ValueError(f"Unknown perturbation mode: {mode!r}")
    return S2, mag


@torch.no_grad()
def generate_one_example(
    nca: SimpleNCA,
    base_pattern: torch.Tensor,
    cfg: WorldConfig,
    dcfg: DatasetConfig,
    rng: torch.Generator,
    example_seed: int,
) -> Dict[str, torch.Tensor]:
    """One (state, perturbation, future outcome) tuple."""
    H, W = cfg.H, cfg.W
    device = cfg.device

    dist = float(
        dcfg.dist_min
        + (dcfg.dist_max - dcfg.dist_min) * torch.rand(1, generator=rng).item()
    )
    center = sample_center_at_distance(dcfg.wells, H, W, dist, rng)
    S0 = place_pattern(base_pattern, center, base_center=BASE_CENTER)

    world = ResourceGrid(
        nca,
        config=cfg,
        S0=S0,
        pattern_center=center,
        seed=example_seed,
    )
    # Burn-in: unperturbed v0 dynamics
    for _ in range(dcfg.burn_in):
        world.step(gate_mode="gated")

    S_t = world.S.detach().clone()
    R_t = world.R.detach().clone()
    # Identity reference for coherence over the horizon: state at query time
    S_ref = S_t.clone()

    # Sample perturbation
    u = float(torch.rand(1, generator=rng).item())
    if u < dcfg.null_frac:
        mode = "null"
    elif u < dcfg.null_frac + 0.5 * (1.0 - dcfg.null_frac):
        mode = "noise"
    else:
        mode = "zero"

    # Prefer probing near pattern mass; fall back to uniform
    mag_map = S_t.norm(dim=1)[0]
    active = (mag_map > DEFAULT_ACTIVATION_THRESHOLD).nonzero(as_tuple=False)
    if active.numel() > 0 and mode != "null":
        idx = int(torch.randint(0, active.shape[0], (1,), generator=rng).item())
        py, px = int(active[idx, 0].item()), int(active[idx, 1].item())
    else:
        py = int(torch.randint(0, H, (1,), generator=rng).item())
        px = int(torch.randint(0, W, (1,), generator=rng).item())

    k = int(dcfg.patch_sizes[int(torch.randint(0, len(dcfg.patch_sizes), (1,), generator=rng).item())])
    S_pert, mag = apply_perturbation(
        S_t, py, px, mode, k, dcfg.noise_std, rng
    )
    world.S = S_pert

    died_t: Optional[int] = None
    for h in range(1, dcfg.horizon + 1):
        world.step(gate_mode="gated")
        if died_t is None and self_mass(world.S) <= DEATH_MASS_EPS:
            died_t = h

    coh = coherence(world.S, S_ref)
    local_r = local_resource_at_pattern_centroid(world.R, world.S)
    died = 1.0 if died_t is not None else 0.0
    # Training-friendly finite targets (NaN → 0; survival is the explicit death bit)
    coh_t = 0.0 if (isinstance(coh, float) and math.isnan(coh)) else float(coh)
    lr_t = 0.0 if (isinstance(local_r, float) and math.isnan(local_r)) else float(local_r)

    pert_map = make_perturbation_map(H, W, py, px, mag, device=device)

    # Store without leading batch dim so stacking yields [N, C, H, W] / [N, 1, H, W]
    return {
        "S": S_t[0].cpu().float(),
        "R": R_t[0].cpu().float(),
        "pert_map": pert_map[0].cpu().float(),
        "pert_y": torch.tensor(py, dtype=torch.int64),
        "pert_x": torch.tensor(px, dtype=torch.int64),
        "pert_mag": torch.tensor(mag, dtype=torch.float32),
        "coherence_future": torch.tensor(coh_t, dtype=torch.float32),
        "local_R_future": torch.tensor(lr_t, dtype=torch.float32),
        "survived": torch.tensor(1.0 - died, dtype=torch.float32),
        "died_within_horizon": torch.tensor(died, dtype=torch.float32),
        "time_of_death": torch.tensor(
            -1 if died_t is None else died_t, dtype=torch.int64
        ),
        "distance_to_well": torch.tensor(
            nearest_well_distance(center[0], center[1], dcfg.wells),
            dtype=torch.float32,
        ),
        "center_y": torch.tensor(center[0], dtype=torch.int64),
        "center_x": torch.tensor(center[1], dtype=torch.int64),
        # raw nan flags for calibration filtering
        "coherence_was_nan": torch.tensor(
            1.0 if (isinstance(coh, float) and math.isnan(coh)) else 0.0
        ),
        "local_R_was_nan": torch.tensor(
            1.0 if (isinstance(local_r, float) and math.isnan(local_r)) else 0.0
        ),
    }


@torch.no_grad()
def generate_dataset(
    dcfg: Optional[DatasetConfig] = None,
    ckpt_path: Path | str = DEFAULT_CKPT,
    out_path: Optional[Path | str] = None,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Generate and optionally save a stacked dataset dict."""
    dcfg = dcfg or DatasetConfig()
    nca, base_pattern, cfg = load_v0_world(ckpt_path, device=dcfg.device)
    cfg = WorldConfig(
        H=cfg.H,
        W=cfg.W,
        channels=cfg.channels,
        hidden_dim=cfg.hidden_dim,
        gen_rate=cfg.gen_rate,
        consumption_rate=cfg.consumption_rate,
        decay_rate=cfg.decay_rate,
        diffusion_rate=cfg.diffusion_rate,
        resource_diffusion=cfg.resource_diffusion,
        well_sigma=cfg.well_sigma,
        wells=dcfg.wells,
        device=dcfg.device,
    )

    rng = torch.Generator(device="cpu")
    rng.manual_seed(dcfg.seed)

    rows: List[Dict[str, torch.Tensor]] = []
    log_every = max(1, dcfg.n_examples // 20)
    for i in range(dcfg.n_examples):
        ex = generate_one_example(
            nca, base_pattern, cfg, dcfg, rng, example_seed=dcfg.seed + 10_000 + i
        )
        rows.append(ex)
        if verbose and ((i + 1) % log_every == 0 or i == 0):
            print(f"  dataset {i + 1}/{dcfg.n_examples}")

    # Stack
    keys = rows[0].keys()
    data = {k: torch.stack([r[k] for r in rows], dim=0) for k in keys}
    data["meta"] = {
        "n_examples": dcfg.n_examples,
        "burn_in": dcfg.burn_in,
        "horizon": dcfg.horizon,
        "seed": dcfg.seed,
        "null_frac": dcfg.null_frac,
        "dist_min": dcfg.dist_min,
        "dist_max": dcfg.dist_max,
        "ckpt": str(ckpt_path),
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        if verbose:
            print(f"  wrote {out_path} ({dcfg.n_examples} examples)")
    return data


def train_val_split(
    data: Dict[str, torch.Tensor],
    val_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    n = data["S"].shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(round(n * val_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    def take(idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = {k: v[idx] for k, v in data.items() if isinstance(v, torch.Tensor)}
        return out

    return take(train_idx), take(val_idx)


class RolloutTensorDataset(torch.utils.data.Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data
        self.n = data["S"].shape[0]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "S": self.data["S"][i],
            "R": self.data["R"][i],
            "pert_map": self.data["pert_map"][i],
            "coherence_future": self.data["coherence_future"][i],
            "local_R_future": self.data["local_R_future"][i],
            "survived": self.data["survived"][i],
        }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate v1 perturbation-rollout dataset")
    p.add_argument("--n", type=int, default=12_000)
    p.add_argument("--out", type=str, default="results/rollout_dataset.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--burn_in", type=int, default=20)
    args = p.parse_args()

    dcfg = DatasetConfig(
        n_examples=args.n,
        horizon=args.horizon,
        burn_in=args.burn_in,
        seed=args.seed,
        device=args.device,
    )
    generate_dataset(dcfg, out_path=Path(args.out), verbose=True)
