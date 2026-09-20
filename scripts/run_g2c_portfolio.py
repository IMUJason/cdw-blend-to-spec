"""Technology-first decomposition prototype (G2c).

Motivation (from the G2 diagnosis): at city scale the monolithic MILP's hardness
is localized in the TECHNOLOGY-SELECTION binaries -- fixing the technology
portfolio collapses the gap from 1.43% to 0.01% and the time from >300s to 112s.
So a portfolio search can act as a primal heuristic while the monolithic solver
supplies a valid dual bound.

Three arms under the SAME total wall-clock budget:
  A  monolithic only
  B  monolithic (for the bound) + portfolio local search (for the incumbent)
  C  same as B, then feed the improved portfolio back as a MIP start

Reported per arm: best feasible objective, best valid bound, relative gap,
number of portfolios priced, and the portfolio's technology pattern.
"""
import json
import time
from pathlib import Path

import numpy as np

from src.models.blend_net import build_gate_instance, literature_composition
from src.models.blend_net import BlendNetworkTwoStage

OUT = Path("results") / "blend"


def tech_pattern(open_dict):
    return tuple(sorted(k for k, v in open_dict.items() if v))


def monolithic(inst, scen, tl, fix=None, mip_start=None):
    m = BlendNetworkTwoStage(inst, scen, mode="saa", landfill_cap_frac=0.45,
                             time_limit=tl, fix_open=fix)
    if mip_start is not None:
        try:
            m.m.add_mip_start(mip_start) if hasattr(m.m, "add_mip_start") else None
        except Exception:
            pass
    t0 = time.time()
    r = m.solve()
    wall = time.time() - t0
    if r is None:
        return None
    return {"objective": r["objective"], "gap": r["gap"],
            "open": tech_pattern(r["open"]), "wall": wall,
            "bound": r["objective"] * (1 - r["gap"])
            if r["gap"] is not None else None}


def price(inst, scen, portfolio, tl):
    m = BlendNetworkTwoStage(inst, scen, mode="saa", landfill_cap_frac=0.45,
                             time_limit=tl, fix_open=portfolio)
    r = m.solve()
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inst = build_gate_instance(seed=11, composition="literature", n_sites=40,
                               n_fac=20, n_dem=6, demand_cover=0.85)
    rng = np.random.default_rng(2024)
    pool = [literature_composition(seed=int(rng.integers(1, 10**6)),
                                   n_sites=40)["brick_share"] for _ in range(10)]
    TOTAL = 300.0
    PRICE_TL = 15.0
    out = {"budget_s": TOTAL, "price_tl_s": PRICE_TL, "arms": {}}

    # ---------------- arm A: monolithic -------------------------------
    t_start = time.time()
    A = monolithic(inst, pool, TOTAL)
    out["arms"]["A_monolithic"] = A
    print(f"[A] monolithic {TOTAL:.0f}s: obj {A['objective']:.0f} "
          f"gap {A['gap']:.4f} wall {A['wall']:.0f}s", flush=True)

    # ---------------- arm B: monolithic + portfolio local search -------
    # Arm B gets its OWN budget clock: half for the bound-producing seed solve,
    # half for portfolio pricing (each pricing is cheap: a fixed portfolio
    # solves to proven optimality in seconds).
    t_b = time.time()
    B_budget = TOTAL
    seed = monolithic(inst, pool, B_budget * 0.5)
    best = dict(seed)
    priced, tried = 0, set()
    history = []
    improved_rounds = 0
    while (time.time() - t_b) < B_budget - PRICE_TL and improved_rounds < 4:
        base = set(best["open"])
        neighbours = []
        for (f, tech) in sorted(base):
            nb = frozenset(base - {(f, tech)} | {(f, 1 - tech)})
            if nb not in tried:
                neighbours.append(nb)
        if not neighbours:
            break
        round_improved = False
        for nb in neighbours:
            if (time.time() - t_b) >= B_budget - PRICE_TL:
                break
            tried.add(nb)
            fix = {}
            for fac in range(inst["n_fac"]):
                for tt in range(inst["n_tech"]):
                    fix[(fac, tt)] = 1 if (fac, tt) in nb else 0
            r = price(inst, pool, fix, PRICE_TL)
            priced += 1
            if r is None:
                continue
            history.append({"portfolio": sorted(nb), "objective": r["objective"],
                            "gap": r["gap"]})
            if r["objective"] < best["objective"]:
                best = {"objective": r["objective"], "gap": r["gap"],
                        "open": tech_pattern(r["open"]), "wall": time.time() - t_b,
                        "bound": None}
                round_improved = True
        improved_rounds = improved_rounds + 1 if round_improved else 4
    out["arms"]["B_monolithic_plus_portfolio"] = {
        "seed_objective": seed["objective"], "seed_gap": seed["gap"],
        "seed_bound": seed["bound"], "best_objective": best["objective"],
        "best_gap_within_portfolio": best["gap"],
        "portfolios_priced": priced, "rounds": improved_rounds,
        "total_wall": time.time() - t_b, "history": history}
    print(f"[B] seed {seed['objective']:.0f} (gap {seed['gap']:.4f}, "
          f"valid bound {seed['bound']:.0f}) -> best {best['objective']:.0f} "
          f"after pricing {priced} portfolios in {time.time()-t_b:.0f}s", flush=True)

    # ---------------- arm C: feed the improved portfolio back ----------
    t0 = time.time()
    fix_best = {}
    for fac in range(inst["n_fac"]):
        for tt in range(inst["n_tech"]):
            fix_best[(fac, tt)] = 1 if (fac, tt) in set(best["open"]) else 0
    C = monolithic(inst, pool, TOTAL, fix=fix_best)
    out["arms"]["C_warm_fixed"] = C
    print(f"[C] warm-fixed portfolio {TOTAL:.0f}s: obj {C['objective']:.0f} "
          f"gap {C['gap']:.4f} wall {C['wall']:.0f}s (own bound)", flush=True)

    with open(OUT / "g2c_portfolio.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nsaved {OUT/'g2c_portfolio.json'}")


if __name__ == "__main__":
    main()
