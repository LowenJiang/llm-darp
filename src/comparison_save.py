"""comparison.py – Compare optimal vs random DARP perturbation across environments.

Scenarios:
  1. Optimal: relaxation + rounding from or_tools_pert.py
  2. Random: uniform random action (0–15) per request

Usage:
    python src/comparison.py [--envs 600] [--customers 30] [--vehicles 5]
                             [--time-limit 5] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from oracle_generator import SFGenerator  # noqa: E402
from optimal_actions import (  # noqa: E402
    FLEX_STR_TO_IDX,
    build_decision_lookup,
    get_accepted_actions,
)
from or_tools_pert import (  # noqa: E402
    _prepare_td,
    darp_solve,
    widen_time_windows,
)


# ===================================================================
# Expanded action space: {-30,-20,-10,0,10,20,30}^2
# ===================================================================
_SHIFT_STEPS_LIST = [-30, -20, -10, 0, 10, 20, 30]
_SHIFT_GRID = np.array(_SHIFT_STEPS_LIST, dtype=float)


def _apply_shifts(td, shift_dict: Dict[int, Tuple[int, int]]):
    """Return a copy of *td* with time windows shifted by raw (ps, ds) pairs."""
    td2 = td.clone()
    tw = td2["time_windows"].clone()
    for r, (ps, ds) in shift_dict.items():
        p, d = 2 * r - 1, 2 * r
        tw[0, p, 0] += ps
        tw[0, p, 1] += ps
        tw[0, d, 0] += ds
        tw[0, d, 1] += ds
    tw = torch.clamp(tw, min=30, max=1410)
    td2["time_windows"] = tw
    return _prepare_td(td2)


def _snap_smallest(lo: float, hi: float) -> int:
    """Pick the discrete step in [lo, hi] with smallest absolute value."""
    valid = _SHIFT_GRID[(_SHIFT_GRID >= lo) & (_SHIFT_GRID <= hi)]
    if len(valid) == 0:
        return 0
    return int(valid[np.argmin(np.abs(valid))])


# ===================================================================
# Retroactive action selection (rounding step)
# ===================================================================

def select_actions_max(
    visit_times: Dict[int, float],
    tw_original: np.ndarray,
    accepted_per_request: Dict[int, List[Tuple[int, int, int]]],
    n_customers: int,
) -> Dict[int, Tuple[int, int]]:
    """Round relaxed visit times to the smallest discrete (ps, ds) shift.

    For each request, the valid range is computed independently:
      ps ∈ [B_p - l_p,  B_p - e_p]
      ds ∈ [B_d - l_d,  B_d - e_d]

    User preferences limit early pickup (ps<0) and late dropoff (ds>0);
    delay pickup (ps>0) and advance dropoff (ds<0) are always permissible.
    """
    selected: Dict[int, Tuple[int, int]] = {}

    for r in range(1, n_customers + 1):
        p, d = 2 * r - 1, 2 * r
        B_p = visit_times.get(p)
        B_d = visit_times.get(d)

        if B_p is None or B_d is None:
            selected[r] = (0, 0)
            continue

        e_p, l_p = float(tw_original[p, 0]), float(tw_original[p, 1])
        e_d, l_d = float(tw_original[d, 0]), float(tw_original[d, 1])

        accepted = accepted_per_request.get(r, [(12, 0, 0)])
        user_min_ps = min(a[1] for a in accepted)   # e.g. -30
        user_max_ds = max(a[2] for a in accepted)   # e.g. +30

        # Valid ps: shifted window [e_p+ps, l_p+ps] must contain B_p
        ps_lo = max(B_p - l_p, user_min_ps)  # early pickup limited by user pref
        ps_hi = B_p - e_p                     # delay pickup always ok

        # Valid ds: shifted window [e_d+ds, l_d+ds] must contain B_d
        ds_lo = B_d - l_d                     # advance dropoff always ok
        ds_hi = min(B_d - e_d, user_max_ds)   # late dropoff limited by user pref

        ps = _snap_smallest(ps_lo, ps_hi)
        ds = _snap_smallest(ds_lo, ds_hi)
        selected[r] = (ps, ds)

    return selected


# ===================================================================
# Random perturbation baseline
# ===================================================================

_SHIFT_STEPS = np.array([-30, -20, -10, 0, 10, 20, 30])
# All 49 (pickup_shift, dropoff_shift) pairs — precomputed once
_ALL_PAIRS = np.array([(ps, ds) for ps in _SHIFT_STEPS for ds in _SHIFT_STEPS])


def random_perturbation(
    td,
    n_customers: int,
    rng: np.random.Generator,
    accepted_per_request: Dict[int, List[Tuple[int, int, int]]] | None = None,
):
    """Apply a uniform-random discrete perturbation per request.

    For each request, enumerate all 49 (ps, ds) pairs from
    {-30,-20,-10,0,10,20,30}^2, keep only those satisfying the
    serviceability constraint (dropoff_early' - pickup_late' >= travel_time + 15),
    and sample one uniformly at random.
    """
    td2 = td.clone()
    tw = td2["time_windows"].clone()
    h3_idx = td["h3_indices"][0].cpu().numpy().astype(int)
    tt_mat = td["travel_time_matrix"][0].cpu().numpy()

    for r in range(1, n_customers + 1):
        p, d = 2 * r - 1, 2 * r
        ep, lp = float(tw[0, p, 0]), float(tw[0, p, 1])
        ed, ld = float(tw[0, d, 0]), float(tw[0, d, 1])
        travel_time = float(tt_mat[h3_idx[p], h3_idx[d]])
        min_gap = travel_time + 15.0

        # Shifted bounds for all 49 pairs (vectorised)
        new_lp = np.minimum(1410.0, lp + _ALL_PAIRS[:, 0])
        new_ed = np.maximum(30.0, ed + _ALL_PAIRS[:, 1])
        feasible = new_ed - new_lp >= min_gap
        feasible_pairs = _ALL_PAIRS[feasible]

        if len(feasible_pairs) == 0:
            continue  # keep original windows

        choice = feasible_pairs[rng.integers(len(feasible_pairs))]
        ps, ds = int(choice[0]), int(choice[1])

        # Customer rejects if early pickup or late dropoff exceeds preference
        if accepted_per_request is not None:
            accepted = accepted_per_request.get(r, [(12, 0, 0)])
            user_min_ps = min(a[1] for a in accepted)  # most negative ps tolerated
            user_max_ds = max(a[2] for a in accepted)  # largest ds tolerated
            if ps < user_min_ps or ds > user_max_ds:
                continue  # customer turns it down → keep original windows

        tw[0, p, 0] = max(30.0, ep + ps)
        tw[0, p, 1] = min(1410.0, lp + ps)
        tw[0, d, 0] = max(30.0, ed + ds)
        tw[0, d, 1] = min(1410.0, ld + ds)

    td2["time_windows"] = tw
    return _prepare_td(td2), None


# ===================================================================
# Main comparison loop
# ===================================================================

def run_comparison(args):
    csv_path = SRC_DIR / "traveler_trip_types_res_7.csv"
    ttm_path = SRC_DIR / "travel_time_matrix_res_7.csv"
    decisions_csv = SRC_DIR / "traveler_decisions_augmented.csv"

    n_cust = args.customers
    n_envs = args.envs

    print("Loading traveler decisions ...")
    decision_lookup = build_decision_lookup(decisions_csv)
    idx_to_flex = {v: k for k, v in FLEX_STR_TO_IDX.items()}

    rng = np.random.default_rng(args.seed)

    optimal_improvements: List[float] = []
    random_improvements: List[float] = []
    skipped = 0

    pbar = tqdm(range(n_envs), desc="Environments", unit="env")
    for env_i in pbar:
        seed_i = args.seed + env_i
        pbar.set_postfix(seed=seed_i, skipped=skipped)

        # --- generate instance ------------------------------------------------
        generator = SFGenerator(
            csv_path=csv_path,
            travel_time_matrix_path=ttm_path,
            num_customers=n_cust,
            perturbation=0,
            seed=seed_i,
        )
        td = generator(batch_size=[1])
        td = _prepare_td(td)

        tw = td["time_windows"][0].cpu().numpy()
        user_ids = td["user_id"][0].cpu().numpy()

        # --- baseline (shared) ------------------------------------------------
        baseline = darp_solve(
            td, max_vehicles=args.vehicles,
            time_limit_seconds=args.time_limit)

        if baseline["status"] == "infeasible" or baseline["total_distance"] <= 0:
            skipped += 1
            continue

        base_cost = baseline["total_distance"]

        # --- OPTIMAL: relaxation + rounding -----------------------------------
        trip_metadata = td["trip_metadata"][0]
        flex_raw = td["flexibility"][0].cpu().numpy()

        accepted_per_request: Dict[int, List[Tuple[int, int, int]]] = {}
        for r in range(1, n_cust + 1):
            p_node = 2 * r - 1
            traveler_id = int(user_ids[p_node])
            flex_idx = int(flex_raw[p_node])
            flex_str = idx_to_flex.get(
                flex_idx, "inflexible for any schedule changes")
            meta = trip_metadata.get(traveler_id)
            if meta is None:
                accepted_per_request[r] = [(12, 0, 0)]
                continue
            accepted_per_request[r] = get_accepted_actions(
                traveler_id, meta, decision_lookup, flex_str)

        td_wide = widen_time_windows(td, accepted_per_request, n_cust)
        slacked = darp_solve(
            td_wide, max_vehicles=args.vehicles,
            time_limit_seconds=args.time_limit)

        if slacked["status"] == "infeasible":
            opt_improvement = 0.0
        else:
            selected = select_actions_max(
                slacked["visit_times"], tw, accepted_per_request, n_cust)
            td_opt = _apply_shifts(td, selected)
            opt_result = darp_solve(
                td_opt, max_vehicles=args.vehicles,
                time_limit_seconds=args.time_limit)
            if opt_result["status"] == "infeasible":
                opt_improvement = 0.0
            else:
                opt_improvement = max(0.0, (
                    (base_cost - opt_result["total_distance"])
                    / base_cost * 100
                ))

        optimal_improvements.append(opt_improvement)

        # --- RANDOM: uniform action per request -------------------------------
        td_rand, _ = random_perturbation(td, n_cust, rng, accepted_per_request)
        rand_result = darp_solve(
            td_rand, max_vehicles=args.vehicles,
            time_limit_seconds=args.time_limit)

        if rand_result["status"] == "infeasible":
            rand_improvement = 0.0
        else:
            rand_improvement = (
                (base_cost - rand_result["total_distance"])
                / base_cost * 100
            )

        random_improvements.append(rand_improvement)

    pbar.close()
    print(f"Done. {n_envs - skipped}/{n_envs} environments solved, "
          f"{skipped} skipped (infeasible baseline).")

    return np.array(optimal_improvements), np.array(random_improvements)


# ===================================================================
# Histogram plot
# ===================================================================

def plot_histogram(optimal_impr, random_impr, save_path):
    all_vals = np.concatenate([optimal_impr, random_impr])
    vmin = np.floor(all_vals.min() / 5) * 5
    vmax = np.ceil(all_vals.max() / 5) * 5
    bins = np.arange(vmin, vmax + 5.01, 5)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(random_impr, bins=bins, alpha=0.55, label="Random Perturbation",
            color="tab:orange", edgecolor="black", linewidth=0.5)
    ax.hist(optimal_impr, bins=bins, alpha=0.55, label="Optimal Perturbation",
            color="tab:blue", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Cost Improvement (%)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title("Perturbation Baseline & Optimal", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved histogram → {save_path}")
    plt.close(fig)


# ===================================================================
# Entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare optimal vs random DARP perturbation")
    parser.add_argument("--envs", type=int, default=600)
    parser.add_argument("--customers", type=int, default=30)
    parser.add_argument("--vehicles", type=int, default=5)
    parser.add_argument("--time-limit", type=int, default=5,
                        help="OR-Tools time limit per solve (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    optimal_impr, random_impr = run_comparison(args)

    if len(optimal_impr) == 0:
        print("\nNo feasible environments — nothing to plot. "
              "Try increasing --vehicles or decreasing --customers.")
        return

    figures_dir = SRC_DIR.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    save_path = figures_dir / "perturbation_comparison_expanded.png"

    plot_histogram(optimal_impr, random_impr, save_path)

    # Summary statistics
    print("\n--- Summary Statistics ---")
    print(f"  Optimal: mean={optimal_impr.mean():.2f}%  "
          f"median={np.median(optimal_impr):.2f}%  "
          f"std={optimal_impr.std():.2f}%")
    print(f"  Random:  mean={random_impr.mean():.2f}%  "
          f"median={np.median(random_impr):.2f}%  "
          f"std={random_impr.std():.2f}%")


if __name__ == "__main__":
    main()
