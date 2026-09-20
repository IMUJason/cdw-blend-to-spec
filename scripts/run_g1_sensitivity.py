"""G1 sensitivity: where does composition heterogeneity actually bite?

Sweeps composition heterogeneity (site-to-site brick-share spread) against the
high-grade price premium, and reports for each configuration:
  - fraction of realizations where the composition-aware optimal PLAN differs
    from the mean-composition plan
  - ex-post regret of the mean-composition (ignorant) plan
  - high-grade service rate (is the quality barrier binding at all?)
  - advanced-sorting adoption rate

Purpose: locate the parameter region in which the blend-to-specification
structure has a substantive story, rather than tuning to a desired answer.
"""
import json
import time
from pathlib import Path

import numpy as np

from src.models.blend_net import (build_gate_instance, BlendNetworkModel,
                                  evaluate_plan)

OUT = Path("results") / "blend"


def plan_key(sol):
    return (tuple(sorted(k for k, v in sol["open"].items() if v)),
            tuple(sorted(k for k, v in sol["batch"].items() if v)))


def run_config(comp_spread, high_price, n_real=30, seed=11):
    inst = build_gate_instance(seed=seed, comp_spread=comp_spread)
    inst["demands"][0]["price"] = high_price
    n = inst["n_sites"]
    b0 = inst["b_base"].copy()
    rng = np.random.default_rng(seed + 100)

    bm = np.full(n, b0.mean())
    sol_mean = BlendNetworkModel(inst, bm, enforce_spec=True).solve()
    if sol_mean is None:
        return None
    key_mean = plan_key(sol_mean)

    rows = []
    for mode in ("permutation", "resample"):
        for _ in range(n_real):
            b = (rng.permutation(b0) if mode == "permutation"
                 else np.clip(rng.normal(b0.mean(), b0.std(), size=n), 0.02, 0.85))
            sol = BlendNetworkModel(inst, b, enforce_spec=True).solve()
            if sol is None:
                continue
            ev = evaluate_plan(inst, b, sol_mean)
            if not ev.get("feasible"):
                continue
            rows.append({
                "plan_differs": plan_key(sol) != key_mean,
                "regret": (ev["realized_cost"] - sol["objective"])
                          / abs(sol["objective"]),
                "output_loss": ev["delivered_tonnage"]
                               < 0.99 * sum(sol["product_d"].values()),
                "served_high": sol["product_d"][0] > 1.0,
                "adv": sum(v for (f, t), v in sol["open"].items() if t == 1) > 0,
            })
    if not rows:
        return None
    g = lambda k: float(np.mean([r[k] for r in rows]))
    reg = np.array([r["regret"] for r in rows])
    return {"comp_spread": comp_spread, "high_price": high_price, "n": len(rows),
            "plan_differs": g("plan_differs"), "regret_mean": float(reg.mean()),
            "regret_p90": float(np.quantile(reg, 0.9)),
            "regret_max": float(reg.max()),
            "output_loss": g("output_loss"), "served_high": g("served_high"),
            "adv_tech": g("adv")}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    for spread in (0.5, 1.0, 2.0):
        for price in (52.0, 64.0, 78.0):
            r = run_config(spread, price)
            if r is None:
                print(f"spread={spread} price={price}: infeasible")
                continue
            res.append(r)
            print(f"spread={spread:4.1f} highprice={price:5.0f}: "
                  f"plan-differs {r['plan_differs']*100:5.1f}% | "
                  f"regret mean {r['regret_mean']*100:5.2f}% "
                  f"p90 {r['regret_p90']*100:5.2f}% max {r['regret_max']*100:5.2f}% | "
                  f"output-loss {r['output_loss']*100:5.1f}% | "
                  f"high-served {r['served_high']*100:5.1f}% | "
                  f"adv {r['adv_tech']*100:5.1f}%", flush=True)
    with open(OUT / "g1_sensitivity.json", "w") as f:
        json.dump({"rows": res, "runtime_s": time.time() - t0}, f, indent=2)
    print(f"\nsaved {OUT/'g1_sensitivity.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
