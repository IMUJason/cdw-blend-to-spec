"""Gate G1 under LITERATURE-calibrated composition (structure type x age x
district stock profile), plus the temporal (urban-renewal) shift variant.

Reports, for the literature-calibrated composition:
  - the realized site-level brick-share spread (is heterogeneity material?)
  - P(optimal plan differs) and the ex-post regret of the mean-composition plan
  - which grades get served and whether advanced sorting is adopted
and repeats the analysis under a temporal shift toward older brick-heavier
stock (a documented urban-renewal pattern), which is the temporal dimension of
composition uncertainty.
"""
import json
import time
from pathlib import Path

import numpy as np

from src.models.blend_net import (build_gate_instance, literature_composition,
                                  BlendNetworkModel, evaluate_plan)

OUT = Path("results") / "blend"


def plan_key(sol):
    return (tuple(sorted(k for k, v in sol["open"].items() if v)),
            tuple(sorted(k for k, v in sol["batch"].items() if v)))


def run(seed, temporal_shift, high_price, n_real=40):
    inst = build_gate_instance(seed=seed, composition="literature")
    inst["demands"][0]["price"] = high_price
    n = inst["n_sites"]
    b0 = inst["b_base"].copy()
    rng_lit = literature_composition(seed=seed, n_sites=n,
                                     temporal_shift=temporal_shift)
    b_mean_vec = np.full(n, b0.mean())

    sol_mean = BlendNetworkModel(inst, b_mean_vec, enforce_spec=True).solve()
    if sol_mean is None:
        return None
    key_mean = plan_key(sol_mean)

    rng = np.random.default_rng(seed + 300)
    rows = []
    for mode in ("resample", "temporal"):
        for _ in range(n_real):
            if mode == "resample":
                # resample district assignments/portfolios at the same epoch
                b = literature_composition(seed=int(rng.integers(1, 10**6)),
                                           n_sites=n)["brick_share"]
            else:
                # a renewal wave shifts the city toward older, brick-heavier
                # stock while total tonnage is unchanged
                b = literature_composition(
                    seed=int(rng.integers(1, 10**6)), n_sites=n,
                    temporal_shift=temporal_shift)["brick_share"]
            sol = BlendNetworkModel(inst, b, enforce_spec=True).solve()
            if sol is None:
                continue
            ev = evaluate_plan(inst, b, sol_mean)
            if not ev.get("feasible"):
                continue
            rows.append({
                "mode": mode,
                "b_std": float(np.std(b)),
                "plan_differs": plan_key(sol) != key_mean,
                "regret": (ev["realized_cost"] - sol["objective"])
                          / abs(sol["objective"]),
                "output_loss": ev["delivered_tonnage"]
                               < 0.99 * sum(sol["product_d"].values()),
                "served_high": sol["product_d"][0] > 1.0,
                "adv_tech": sum(v for (f, t), v in sol["open"].items()
                                if t == 1) > 0,
            })
    if not rows:
        return None
    g = lambda k: float(np.mean([r[k] for r in rows]))
    reg = np.array([r["regret"] for r in rows])
    return {
        "seed": seed, "temporal_shift": temporal_shift, "high_price": high_price,
        "n": len(rows),
        "b_std_mean": g("b_std"),
        "plan_differs": g("plan_differs"),
        "regret_mean": float(reg.mean()),
        "regret_p90": float(np.quantile(reg, 0.9)),
        "regret_max": float(reg.max()),
        "output_loss": g("output_loss"),
        "served_high": g("served_high"),
        "adv_tech": g("adv_tech"),
        "by_mode": {m: {"n": len([r for r in rows if r["mode"] == m]),
                        "plan_differs": float(np.mean(
                            [r["plan_differs"] for r in rows if r["mode"] == m])),
                        "regret_mean": float(np.mean(
                            [r["regret"] for r in rows if r["mode"] == m])),
                        "regret_max": float(np.max(
                            [r["regret"] for r in rows if r["mode"] == m]))}
                    for m in ("resample", "temporal")},
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    lit = literature_composition(seed=11, n_sites=8)
    print("district assignment:", lit["profile_assignment"])
    print("literature-calibrated brick shares:",
          np.round(lit["brick_share"], 3),
          f"| mean {lit['mean']:.3f} std {lit['std']:.3f}")
    print()

    res = []
    for shift in (0.0, 0.6):
        for price in (52.0, 70.0):
            r = run(11, shift, price)
            if r is None:
                continue
            res.append(r)
            print(f"renewal-shift={shift:3.1f} highprice={price:4.0f}: "
                  f"b_std {r['b_std_mean']:.3f} | plan-differs "
                  f"{r['plan_differs']*100:5.1f}% | regret mean "
                  f"{r['regret_mean']*100:5.2f}% p90 {r['regret_p90']*100:5.2f}% "
                  f"max {r['regret_max']*100:5.2f}% | output-loss "
                  f"{r['output_loss']*100:5.1f}% | high-served "
                  f"{r['served_high']*100:5.1f}% | adv {r['adv_tech']*100:5.1f}%")
    with open(OUT / "g1_literature.json", "w") as f:
        json.dump({"calibration": {"archetypes": lit["archetypes"],
                                   "profiles": lit["profiles"],
                                   "assignment": lit["profile_assignment"],
                                   "mean": lit["mean"], "std": lit["std"]},
                   "rows": res, "runtime_s": time.time() - t0}, f, indent=2)
    print(f"\nsaved {OUT/'g1_literature.json'} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
