"""
action_selection.py
===================
Candidate generation, validity-masked selection, and the scarce attention
budget mechanic for v2.

No RL / policy gradients — selection is argmax over self_model_v2's predicted
coherence_future under a boost-at-candidate encoding.

Does not modify R physics or SimpleNCA; attention A is an independent scalar.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

V2_ROOT = Path(__file__).resolve().parent
REPO_ROOT = V2_ROOT.parent
V0_ROOT = REPO_ROOT / "v0"
V1_ROOT = REPO_ROOT / "v1"
for p in (str(V0_ROOT), str(V1_ROOT), str(V2_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from boost_rollout_dataset import (  # noqa: E402
    make_boost_map,
    make_perturbation_map,
    step_with_boost,
)
from grid_world import DEFAULT_WELLS, ResourceGrid, WorldConfig  # noqa: E402
from identity_weight import validity_mask  # noqa: E402
from metrics import (  # noqa: E402
    DEATH_MASS_EPS,
    DEFAULT_ACTIVATION_THRESHOLD,
    coherence,
    local_resource_at_pattern_centroid,
    self_mass,
)
from self_model import SelfModel  # noqa: E402
from self_model_v2 import predict_coherence_future  # noqa: E402


@dataclass
class AttentionConfig:
    """Scarce attention budget parameters."""

    K: int = 20  # decision interval (= typical boost duration)
    n_candidates: int = 6  # 4–8 angular sectors
    boost_magnitude: float = 0.40
    boost_duration: int = 20
    boost_sigma: float = 1.5
    boost_cost: float = 1.0
    regen_rate: float = 0.035  # per step; tuned so A cannot cover every candidate
    A_max: float = 1.5
    A0: float = 1.0  # start with enough for one boost
    mass_threshold: float = DEFAULT_ACTIVATION_THRESHOLD
    wells: Tuple[Tuple[int, int], ...] = DEFAULT_WELLS
    # Encoding magnitude for self-model queries (match training boost mag range)
    query_magnitude: Optional[float] = None  # default = boost_magnitude


@dataclass
class Candidate:
    y: int
    x: int
    sector: int
    mass: float
    masked: bool = False  # True if dropped by validity_mask


@dataclass
class SelectionLog:
    """Per-decision-interval bookkeeping for Check 3 / diagnostics."""

    t: int
    policy: str
    n_raw: int
    n_surviving: int
    n_dropped_mask: int
    selected: Optional[Tuple[int, int]]
    selected_sector: Optional[int]  # angular sector id of chosen candidate
    selected_was_masked: bool
    scores: List[float]
    score_of_selected: Optional[float]
    A_before: float
    A_after: float
    boosted: bool


def mass_weighted_centroid(
    S: torch.Tensor,
    threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> Optional[Tuple[float, float]]:
    """Mass-weighted centroid of active support; None if empty."""
    if S.dim() == 4:
        mag = S.norm(dim=1)[0]
    elif S.dim() == 3:
        mag = S.norm(dim=0)
    else:
        raise ValueError(f"S shape {S.shape}")
    mask = mag > threshold
    if not mask.any():
        return None
    ys, xs = torch.where(mask)
    w = mag[mask]
    cy = float((ys.float() * w).sum() / w.sum())
    cx = float((xs.float() * w).sum() / w.sum())
    return cy, cx


def generate_candidates(
    S: torch.Tensor,
    n_candidates: int = 6,
    mass_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> List[Candidate]:
    """
    Partition the pattern's active-mass support into `n_candidates` contiguous
    angular sectors around the mass-weighted centroid. Each candidate is the
    mass-weighted centroid of cells in that sector (integer grid cell).

    Deterministic given S; empty sectors are skipped.
    """
    if S.dim() == 4:
        mag = S.norm(dim=1)[0]
    else:
        mag = S.norm(dim=0)
    H, W = mag.shape
    mask = mag > mass_threshold
    if not mask.any():
        return []

    ys, xs = torch.where(mask)
    w = mag[mask]
    cy = float((ys.float() * w).sum() / w.sum())
    cx = float((xs.float() * w).sum() / w.sum())

    # Angle of each active cell relative to centroid
    dy = ys.float() - cy
    dx = xs.float() - cx
    angles = torch.atan2(dy, dx)  # [-pi, pi]
    # Map to [0, 2pi)
    angles = torch.where(angles < 0, angles + 2.0 * math.pi, angles)
    sector_w = 2.0 * math.pi / n_candidates
    sector_ids = (angles / sector_w).long().clamp(0, n_candidates - 1)

    cands: List[Candidate] = []
    for s in range(n_candidates):
        m = sector_ids == s
        if not m.any():
            continue
        sw = w[m]
        sy = float((ys[m].float() * sw).sum() / sw.sum())
        sx = float((xs[m].float() * sw).sum() / sw.sum())
        iy = int(round(sy))
        ix = int(round(sx))
        iy = max(0, min(H - 1, iy))
        ix = max(0, min(W - 1, ix))
        cands.append(
            Candidate(
                y=iy,
                x=ix,
                sector=s,
                mass=float(sw.sum().item()),
                masked=False,
            )
        )
    return cands


def mask_candidates(
    candidates: List[Candidate],
    S: torch.Tensor,
    wells: Sequence[Tuple[int, int]] = DEFAULT_WELLS,
    mass_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> Tuple[List[Candidate], List[Candidate], int]:
    """
    Apply v1 validity_mask (support AND not-equidistant-boundary) **before**
    scoring. Returns (surviving, dropped, n_dropped).
    """
    vmask = validity_mask(S, mass_threshold=mass_threshold, wells=wells)
    surviving: List[Candidate] = []
    dropped: List[Candidate] = []
    for c in candidates:
        if bool(vmask[c.y, c.x].item()):
            surviving.append(c)
        else:
            c.masked = True
            dropped.append(c)
    return surviving, dropped, len(dropped)


@torch.no_grad()
def score_candidates(
    model: SelfModel,
    S: torch.Tensor,
    R: torch.Tensor,
    candidates: List[Candidate],
    magnitude: float,
    device: str,
    sigma: float = 1.5,
) -> List[float]:
    """
    Self-model predicted coherence_future under boost-at-c encoding.

    Boost queries use a *negative* Gaussian bump (signed magnitude) so the
    encoding is distinguishable from damage-type probes (non-negative).
    `magnitude` is the absolute boost strength; sign is applied here.
    """
    if not candidates:
        return []
    if S.dim() == 3:
        S = S.unsqueeze(0)
    if R.dim() == 3:
        R = R.unsqueeze(0)
    if R.dim() == 2:
        R = R.view(1, 1, *R.shape)
    _, _, H, W = S.shape
    abs_mag = abs(float(magnitude))
    maps = []
    for c in candidates:
        p = make_perturbation_map(
            H, W, c.y, c.x, abs_mag, device=device, sigma=sigma
        )
        maps.append(-p)  # signed: negative ⇔ boost
    P = torch.cat(maps, dim=0)
    B = len(candidates)
    S_b = S.expand(B, -1, -1, -1).contiguous()
    R_b = R.expand(B, -1, -1, -1).contiguous()
    pred = predict_coherence_future(model, S_b, R_b, P, device=device)
    return [float(pred[i].item()) for i in range(B)]


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in radians."""
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return abs(d)


def select_wellward_candidate(
    candidates: List[Candidate],
    S: torch.Tensor,
    wells: Sequence[Tuple[int, int]],
    mass_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> Optional[Candidate]:
    """
    Hand-coded zero-learning heuristic: boost the candidate whose direction
    from the pattern centroid is closest to the nearest well.

    No self-model, no learned weights — pure geometry from well coordinates
    and mass-weighted centroid at runtime.
    """
    if not candidates:
        return None
    cent = mass_weighted_centroid(S, threshold=mass_threshold)
    if cent is None:
        # Fallback: lowest sector id
        return min(candidates, key=lambda c: c.sector)
    cy, cx = cent
    best_w = min(wells, key=lambda w: (w[0] - cy) ** 2 + (w[1] - cx) ** 2)
    well_angle = math.atan2(best_w[0] - cy, best_w[1] - cx)

    def score(c: Candidate) -> float:
        return _angle_diff(math.atan2(c.y - cy, c.x - cx), well_angle)

    return min(candidates, key=score)


def select_candidate(
    candidates: List[Candidate],
    scores: List[float],
    policy: str,
    rng: torch.Generator,
    rr_index: int = 0,
    fixed_sector: Optional[int] = None,
    S: Optional[torch.Tensor] = None,
    wells: Optional[Sequence[Tuple[int, int]]] = None,
    mass_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> Tuple[Optional[Candidate], int, Optional[int]]:
    """
    policy:
      - "selected": argmax scores (self-model driven)
      - "random": uniform among candidates
      - "round_robin": cycle by sector order (rr_index)
      - "fixed": lock one sector at first decision; always re-select that sector
        (or nearest available by sector-id distance if empty). No model.
      - "wellward_heuristic": pick candidate closest in angle to nearest well
        (pure geometry, no model). The 3-line rule that may fully explain selected.

    Returns (chosen, new_rr_index, new_fixed_sector).
    """
    if not candidates:
        return None, rr_index, fixed_sector
    if policy == "selected":
        best_i = int(max(range(len(scores)), key=lambda i: scores[i]))
        return candidates[best_i], rr_index, fixed_sector
    if policy == "random":
        i = int(torch.randint(0, len(candidates), (1,), generator=rng).item())
        return candidates[i], rr_index, fixed_sector
    if policy in ("round_robin", "round-robin"):
        i = rr_index % len(candidates)
        return candidates[i], rr_index + 1, fixed_sector
    if policy in ("wellward_heuristic", "wellward"):
        if S is None or wells is None:
            raise ValueError("wellward_heuristic requires S and wells")
        chosen = select_wellward_candidate(
            candidates, S, wells, mass_threshold=mass_threshold
        )
        return chosen, rr_index, fixed_sector
    if policy == "fixed":
        # Lock a sector id for the whole episode. If fixed_sector is already set
        # (preset or first decision), hold it. Else default: lowest surviving id
        # (legacy single-draw arbitrary baseline).
        if fixed_sector is None:
            fixed_sector = int(min(c.sector for c in candidates))
        exact = [c for c in candidates if c.sector == fixed_sector]
        if exact:
            return exact[0], rr_index, fixed_sector
        # Sector empty this interval: nearest by circular distance mod 6
        mod = 6

        def circ_dist(a: int, b: int, m: int) -> int:
            d = abs(a - b) % m
            return min(d, m - d)

        best = min(
            candidates,
            key=lambda c: (circ_dist(c.sector, fixed_sector, mod), c.sector),
        )
        return best, rr_index, fixed_sector
    raise ValueError(f"Unknown policy: {policy!r}")


def sector_of_angle(angle: float, n_candidates: int) -> int:
    """Map atan2 angle in [-pi, pi] to sector id (same binning as generate_candidates)."""
    a = angle if angle >= 0 else angle + 2.0 * math.pi
    sector_w = 2.0 * math.pi / n_candidates
    return int(a / sector_w) % n_candidates


def wellward_sector(
    centroid: Tuple[float, float],
    well: Tuple[int, int],
    n_candidates: int,
) -> int:
    """Sector pointing from pattern centroid toward a well."""
    cy, cx = centroid
    wy, wx = well
    return sector_of_angle(math.atan2(wy - cy, wx - cx), n_candidates)


def geometry_of_choice(
    S: torch.Tensor,
    chosen: Candidate,
    wells: Sequence[Tuple[int, int]],
    n_candidates: int,
    mass_threshold: float = DEFAULT_ACTIVATION_THRESHOLD,
) -> Dict[str, float]:
    """
    Relate a chosen sector to well geometry.

    Reports centroid, angle of chosen cell from centroid, nearest-well angle,
    wellward sector id, and whether chosen.sector == wellward sector.
    """
    cent = mass_weighted_centroid(S, threshold=mass_threshold)
    if cent is None:
        return {
            "centroid_y": float("nan"),
            "centroid_x": float("nan"),
            "choice_angle": float("nan"),
            "nearest_well_y": float("nan"),
            "nearest_well_x": float("nan"),
            "well_angle": float("nan"),
            "wellward_sector": float("nan"),
            "choice_is_wellward": 0.0,
            "dist_to_nearest_well": float("nan"),
            "angle_error_to_well": float("nan"),
        }
    cy, cx = cent
    choice_angle = math.atan2(chosen.y - cy, chosen.x - cx)
    # Nearest well
    best_w = min(wells, key=lambda w: (w[0] - cy) ** 2 + (w[1] - cx) ** 2)
    well_angle = math.atan2(best_w[0] - cy, best_w[1] - cx)
    ww_sec = wellward_sector(cent, best_w, n_candidates)
    # Angular error in [-pi, pi]
    err = choice_angle - well_angle
    while err > math.pi:
        err -= 2.0 * math.pi
    while err < -math.pi:
        err += 2.0 * math.pi
    return {
        "centroid_y": cy,
        "centroid_x": cx,
        "choice_angle": choice_angle,
        "nearest_well_y": float(best_w[0]),
        "nearest_well_x": float(best_w[1]),
        "well_angle": well_angle,
        "wellward_sector": float(ww_sec),
        "choice_is_wellward": 1.0 if chosen.sector == ww_sec else 0.0,
        "dist_to_nearest_well": float(math.hypot(best_w[0] - cy, best_w[1] - cx)),
        "angle_error_to_well": err,
    }


class AttentionController:
    """
    Independent attention scalar A + decision-interval boost application.

    A regenerates each step; a boost costs boost_cost and lasts boost_duration
    steps. Choosing well matters because A cannot cover every candidate every
    interval under the tuned (regen_rate, boost_cost) operating point.
    """

    def __init__(
        self,
        cfg: Optional[AttentionConfig] = None,
        fixed_sector_preset: Optional[int] = None,
    ):
        self.cfg = cfg or AttentionConfig()
        self.A = float(self.cfg.A0)
        self.boost_remaining = 0
        self.active_boost_map: Optional[torch.Tensor] = None
        self.rr_index = 0
        # If preset given, lock that sector id from t=0 (6-way sweep).
        # Else None → first decision picks lowest surviving id.
        self.fixed_sector: Optional[int] = (
            int(fixed_sector_preset) if fixed_sector_preset is not None else None
        )
        self.logs: List[SelectionLog] = []
        self.total_masked_selected = 0  # must stay 0 (Check 3)
        self.total_candidates_dropped = 0
        self.total_boosts = 0
        self.total_decisions = 0
        self.total_skipped_budget = 0
        self.geometry_logs: List[Dict] = []  # per-decision well/sector geometry

    def regen_step(self) -> None:
        self.A = min(self.cfg.A_max, self.A + self.cfg.regen_rate)

    def tick_boost(self) -> None:
        if self.boost_remaining > 0:
            self.boost_remaining -= 1
            if self.boost_remaining <= 0:
                self.active_boost_map = None

    def maybe_decide(
        self,
        t: int,
        S: torch.Tensor,
        R: torch.Tensor,
        model: Optional[SelfModel],
        policy: str,
        device: str,
        rng: torch.Generator,
    ) -> SelectionLog:
        """
        Called every step; only acts on decision intervals (t % K == 0, t>0)
        or t==0. When budget allows, selects and arms a boost.
        """
        cfg = self.cfg
        is_decision = (t == 0) or (t % cfg.K == 0)
        log = SelectionLog(
            t=t,
            policy=policy,
            n_raw=0,
            n_surviving=0,
            n_dropped_mask=0,
            selected=None,
            selected_sector=None,
            selected_was_masked=False,
            scores=[],
            score_of_selected=None,
            A_before=self.A,
            A_after=self.A,
            boosted=False,
        )
        if not is_decision:
            return log

        self.total_decisions += 1
        raw = generate_candidates(
            S, n_candidates=cfg.n_candidates, mass_threshold=cfg.mass_threshold
        )
        surviving, dropped, n_drop = mask_candidates(
            raw, S, wells=cfg.wells, mass_threshold=cfg.mass_threshold
        )
        self.total_candidates_dropped += n_drop
        log.n_raw = len(raw)
        log.n_surviving = len(surviving)
        log.n_dropped_mask = n_drop
        if n_drop > 0:
            # Rare: pattern support on equidistant boundary — log explicitly
            print(
                f"  [attention t={t}] dropped {n_drop}/{len(raw)} candidates "
                f"via validity_mask (equidistant/support)"
            )

        if not surviving:
            log.A_after = self.A
            self.logs.append(log)
            return log

        qmag = cfg.query_magnitude if cfg.query_magnitude is not None else cfg.boost_magnitude
        if policy == "selected":
            if model is None:
                raise ValueError("policy='selected' requires a self-model")
            scores = score_candidates(
                model, S, R, surviving, magnitude=qmag, device=device, sigma=cfg.boost_sigma
            )
        else:
            # Baselines don't need scores; fill zeros for logging
            scores = [0.0] * len(surviving)
        log.scores = scores

        chosen, self.rr_index, self.fixed_sector = select_candidate(
            surviving,
            scores,
            policy=policy,
            rng=rng,
            rr_index=self.rr_index,
            fixed_sector=self.fixed_sector,
            S=S,
            wells=cfg.wells,
            mass_threshold=cfg.mass_threshold,
        )

        # Check 3: selected must never be a masked-out candidate
        if chosen is not None and chosen.masked:
            self.total_masked_selected += 1
            log.selected_was_masked = True

        if chosen is not None:
            log.selected = (chosen.y, chosen.x)
            log.selected_sector = int(chosen.sector)
            # score index matches surviving order
            try:
                ci = next(
                    i
                    for i, c in enumerate(surviving)
                    if c.y == chosen.y and c.x == chosen.x and c.sector == chosen.sector
                )
                log.score_of_selected = float(scores[ci])
            except StopIteration:
                log.score_of_selected = None
            # Geometry: is this the wellward sector?
            geom = geometry_of_choice(
                S,
                chosen,
                wells=cfg.wells,
                n_candidates=cfg.n_candidates,
                mass_threshold=cfg.mass_threshold,
            )
            geom["t"] = t
            geom["policy"] = policy
            geom["chosen_sector"] = int(chosen.sector)
            self.geometry_logs.append(geom)

        if chosen is not None and self.A >= cfg.boost_cost:
            H, W = S.shape[-2], S.shape[-1]
            self.active_boost_map = make_boost_map(
                H,
                W,
                chosen.y,
                chosen.x,
                cfg.boost_magnitude,
                device=S.device,
                sigma=cfg.boost_sigma,
            )
            self.boost_remaining = cfg.boost_duration
            self.A -= cfg.boost_cost
            log.boosted = True
            self.total_boosts += 1
        else:
            self.total_skipped_budget += 1

        log.A_after = self.A
        self.logs.append(log)
        return log


@torch.no_grad()
def run_attention_episode(
    nca,
    base_pattern: torch.Tensor,
    world_cfg: WorldConfig,
    model: Optional[SelfModel],
    policy: str,
    distance: int,
    horizon: int = 2000,
    well: Tuple[int, int] = (8, 8),
    attn_cfg: Optional[AttentionConfig] = None,
    seed: int = 0,
    device: str = "cpu",
    place_pattern_fn=None,
    base_center: Tuple[int, int] = (16, 16),
    fixed_sector_preset: Optional[int] = None,
) -> Dict:
    """
    Full attention/action loop over `horizon` steps.

    policy in {"selected", "random", "round_robin", "fixed", "wellward_heuristic"}.
      - fixed: lock a sector forever (no model). If fixed_sector_preset is set,
        lock that id; else lock lowest-id survivor at first decision.
      - wellward_heuristic: boost sector nearest in angle to nearest well
        (pure geometry, no model).
    Returns history + selection diagnostics.
    """
    from grid_world import place_pattern as _place

    place = place_pattern_fn or _place
    attn_cfg = attn_cfg or AttentionConfig()
    center = (well[0], well[1] + distance)
    S0 = place(base_pattern, center, base_center=base_center)
    world = ResourceGrid(
        nca, config=world_cfg, S0=S0, pattern_center=center, seed=seed
    )
    ctrl = AttentionController(attn_cfg, fixed_sector_preset=fixed_sector_preset)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed + 1009)

    S_ref = world.S.detach().clone()
    history: List[Dict[str, float]] = []

    def record(t: int, mag_mean: float = 0.0) -> None:
        mass = self_mass(world.S)
        coh = coherence(world.S, S_ref, mass=mass)
        lr = local_resource_at_pattern_centroid(world.R, world.S)
        history.append(
            {
                "t": float(t),
                "self_mass": mass,
                "coherence": (
                    float("nan")
                    if (isinstance(coh, float) and math.isnan(coh))
                    else float(coh)
                ),
                "local_R": (
                    float("nan")
                    if (isinstance(lr, float) and math.isnan(lr))
                    else float(lr)
                ),
                "A": ctrl.A,
                "boost_remaining": float(ctrl.boost_remaining),
                "alive": 1.0 if mass > DEATH_MASS_EPS else 0.0,
            }
        )

    record(0)
    # Decision at t=0 before first step
    ctrl.maybe_decide(0, world.S, world.R, model, policy, device, rng)

    for t in range(1, horizon + 1):
        ctrl.regen_step()
        # Apply active boost via step_with_boost; else plain step
        if ctrl.active_boost_map is not None and ctrl.boost_remaining > 0:
            _, mag = step_with_boost(
                world, boost_map=ctrl.active_boost_map, gate_mode="gated"
            )
        else:
            _, mag = world.step(gate_mode="gated")
        ctrl.tick_boost()
        # Decision intervals after the step at t
        if t % attn_cfg.K == 0:
            ctrl.maybe_decide(t, world.S, world.R, model, policy, device, rng)
        record(t, mag_mean=float(mag.mean().item()))

        if history[-1]["alive"] < 0.5 and t > 50:
            # Pattern dead — continue recording NaNs for full horizon consistency
            # but no need to decide; still advance A for accounting
            pass

    # Summary stats over alive steps
    alive_coh = [
        h["coherence"]
        for h in history
        if h["alive"] > 0.5 and not math.isnan(h["coherence"])
    ]
    alive_mass = [h["self_mass"] for h in history if h["alive"] > 0.5]
    final = history[-1]
    tod = next((int(h["t"]) for h in history if h["alive"] < 0.5), None)

    return {
        "history": history,
        "policy": policy,
        "distance": distance,
        "seed": seed,
        "mean_coherence": float(sum(alive_coh) / len(alive_coh)) if alive_coh else float("nan"),
        "final_coherence": final["coherence"],
        "mean_self_mass": float(sum(alive_mass) / len(alive_mass)) if alive_mass else 0.0,
        "final_self_mass": final["self_mass"],
        "time_of_death": tod,
        "n_boosts": ctrl.total_boosts,
        "n_decisions": ctrl.total_decisions,
        "n_skipped_budget": ctrl.total_skipped_budget,
        "n_candidates_dropped_mask": ctrl.total_candidates_dropped,
        "n_masked_selected": ctrl.total_masked_selected,
        "selection_logs": ctrl.logs,
        "geometry_logs": ctrl.geometry_logs,
        "fixed_sector_locked": ctrl.fixed_sector,
        "attn_cfg": {
            "K": attn_cfg.K,
            "boost_cost": attn_cfg.boost_cost,
            "regen_rate": attn_cfg.regen_rate,
            "A_max": attn_cfg.A_max,
            "boost_magnitude": attn_cfg.boost_magnitude,
            "n_candidates": attn_cfg.n_candidates,
        },
    }


def attention_budget_stats(
    regen_rate: float,
    boost_cost: float,
    K: int,
    A_max: float,
    A0: float = 1.0,
    horizon: int = 2000,
) -> Dict[str, float]:
    """
    Closed-form / simulated budget scarcity metrics (no world needed).

    max_boosts_if_always: how many boosts if we always spend when able
    fractions: expected boosts per decision interval under greedy spend
    """
    A = A0
    boosts = 0
    decisions = 0
    for t in range(0, horizon + 1):
        if t > 0:
            A = min(A_max, A + regen_rate)
        if t % K == 0:
            decisions += 1
            if A >= boost_cost:
                A -= boost_cost
                boosts += 1
    return {
        "regen_rate": regen_rate,
        "boost_cost": boost_cost,
        "K": K,
        "A_max": A_max,
        "horizon": horizon,
        "n_decisions": decisions,
        "n_boosts_greedy": boosts,
        "boost_fraction": boosts / max(1, decisions),
        "regen_per_interval": regen_rate * K,
        "can_boost_every_interval": (regen_rate * K) >= boost_cost,
        "scarce": (regen_rate * K) < boost_cost,
    }


def sweep_attention_budget(
    K: int = 20,
    horizon: int = 2000,
    candidates: Optional[List[Tuple[float, float, float]]] = None,
) -> List[Dict[str, float]]:
    """
    Sweep (regen_rate, boost_cost, A_max). Prefer operating points where
    boost_fraction is in ~0.3–0.7 (scarce but usable).
    """
    if candidates is None:
        candidates = []
        for regen in (0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06):
            for cost in (0.8, 1.0, 1.2):
                for amax in (1.0, 1.5, 2.0):
                    candidates.append((regen, cost, amax))
    rows = []
    for regen, cost, amax in candidates:
        st = attention_budget_stats(
            regen_rate=regen,
            boost_cost=cost,
            K=K,
            A_max=amax,
            A0=min(1.0, amax),
            horizon=horizon,
        )
        rows.append(st)
    return rows


def pick_operating_point(
    sweep_rows: List[Dict[str, float]],
    target_frac: float = 0.5,
) -> Dict[str, float]:
    """Pick scarce operating point closest to target boost fraction."""
    scarce = [r for r in sweep_rows if r["scarce"]]
    pool = scarce if scarce else sweep_rows
    best = min(pool, key=lambda r: abs(r["boost_fraction"] - target_frac))
    return best
