"""G2b: does formulation strengthening remove the city-scale hardness?

Compares, at city scale and equal wall-clock limits:
  baseline      unmodified big-M formulation
  tightened     per-line/per-grade minimal big-M + a-priori infeasible batch
                pruning (w[f,t,d] = 0 when (1-rho_t) min_s b_s > kappa_d)

Also reports the LP-relaxation bound gap, which shows whether the difficulty
comes from a weak relaxation (fixable by strengthening) or from combinatorial
structure (needing decomposition).
"""
import json
import time
from pathlib import Path

import numpy as np

from src.models.blend_net import build_gate_instance, literature_composition
from src.models.blend_net import BlendNetworkTwoStage

OUT = Path("results") / "blend"


def best_bound_and_gap(model):
    """Solver's own best bound and relative gap at the time limit (the bound
    already reflects root cuts, so it measures how hard the instance is for the
    solver rather than the raw LP relaxation)."""
    try:
        d = model.m.solve_details
        bb = getattr(d, "best_bound", None)
        gap = getattr(d, "mip_relative_gap", None)
        return bb, gap
    except Exception:
        return None, None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inst = build_gate_instance(seed=11, composition="literature",
                               n_sites=40, n_fac=20, n_dem=6,
                               demand_cover=0.85)
    rng = np.random.default_rng(2024)
    pool = [literature_composition(seed=int(rng.integers(1, 10**6)),
                                   n_sites=40)["brick_share"]
            for _ in range(25)]
    print(f"city instance: waste {inst['Q'].sum():.0f} t, "
          f"demand {sum(d['quantity'] for d in inst['demands']):.0f} t\n",
          flush=True)

    rows = []
    for K in (10, 25):
        scen = pool[:K]
        for tag, tm, pb in (("baseline", False, False),
                            ("tightened", True, True)):
            m = BlendNetworkTwoStage(inst, scen, mode="saa",
                                     landfill_cap_frac=0.45, time_limit=300,
                                     tight_m=tm, prune_batches=pb)
            nv = m.m.statistics.number_of_variables
            nb = m.m.statistics.number_of_binary_variables
            nc = m.m.statistics.number_of_constraints
            t0 = time.time()
            r = m.solve()
            wall = time.time() - t0
            bb, gap_final = best_bound_and_gap(m)
            rec = {"K": K, "variant": tag, "n_vars": nv, "n_binaries": nb,
                   "n_constraints": nc, "wall_s": wall,
                   "objective": None if r is None else r["objective"],
                   "solver_gap": None if r is None else r["gap"],
                   "best_bound": bb, "gap_reported": gap_final}
            rows.append(rec)
            print(f"K={K:3d} {tag:10s}: {nv:7d} vars / {nb:6d} bin / "
                  f"{nc:7d} cons | wall {wall:6.1f}s | "
                  f"obj {'--' if r is None else round(r['objective'])} "
                  f"gap {0 if r is None else r['gap']:.4f} | "
                  f"bound {('--' if bb is None else round(float(bb)))}",
                  flush=True)

    with open(OUT / "g2_strengthening.json", "w") as f:
        json.dump({"instance": {"n_sites": 40, "n_fac": 20, "n_dem": 6},
                   "rows": rows}, f, indent=2)
    print(f"\nsaved {OUT/'g2_strengthening.json'}")


if __name__ == "__main__":
    main()
