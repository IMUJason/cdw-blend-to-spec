"""G1 boundary v2: finer grid, larger samples, continuous metrics.

Replaces the coarse 3x3 sweep whose binary plan-change rate was dominated by
discrete switching near the certification margin. Reports, per spread level:
  cert_share  share of (site, grade, tech) pairs certifiable at the realized
              composition -- a continuous composition-sensitivity index
  plan_differs, regret_mean, regret_q90, regret_max  as before, n_real=40
"""
import json
from pathlib import Path

import numpy as np

from src.models.blend_net import BlendNetworkModel, build_gate_instance, evaluate_plan

OUT = Path("results") / "blend"


def plan_key(sol):
    return (tuple(sorted(k for k, v in sol["open"].items() if v)),
            tuple(sorted(k for k, v in sol["batch"].items() if v)))


def cert_share(inst, b):
    S, T, D = inst["n_sites"], inst["n_tech"], inst["n_dem"]
    ok = tot = 0
    for t in range(T):
        eta, rho = inst["techs"][t]["yield"], inst["techs"][t]["rho"]
        for d in range(D):
            kap = inst["demands"][d]["kappa"]
            for s in range(S):
                tot += 1
                if (1 - rho) * b[s] <= kap * eta:
                    ok += 1
    return ok / tot


def run(spread, n_real=40, seed=11):
    inst = build_gate_instance(seed=seed, comp_spread=spread)
    b0 = inst["b_base"].copy()
    bm = np.full(inst["n_sites"], b0.mean())
    sol_mean = BlendNetworkModel(inst, bm, enforce_spec=True).solve()
    key_mean = plan_key(sol_mean)
    rng = np.random.default_rng(seed + 100)
    diffs, regrets = [], []
    cs = []
    for _ in range(n_real):
        b = (rng.permutation(b0) if _ % 2 == 0 else
             np.clip(rng.normal(b0.mean(), b0.std(), size=inst["n_sites"]),
                     0.02, 0.85))
        sol = BlendNetworkModel(inst, b, enforce_spec=True).solve()
        ev = evaluate_plan(inst, b, sol_mean)
        if sol is None or not ev.get("feasible"):
            continue
        diffs.append(plan_key(sol) != key_mean)
        regrets.append((ev["realized_cost"] - sol["objective"])
                       / abs(sol["objective"]))
        cs.append(cert_share(inst, b))
    r = np.array(regrets)
    return {"spread": spread, "std": float(b0.std()),
            "cert_share": float(np.mean(cs)),
            "cert_share_mean_comp": cert_share(inst, bm),
            "plan_differs": float(np.mean(diffs)),
            "regret_mean": float(r.mean()), "regret_q90": float(np.quantile(r, .9)),
            "regret_max": float(r.max()), "n": len(regrets)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for spread in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        r = run(spread)
        rows.append(r)
        print(f"spread {spread:4.2f} (std {r['std']:.3f}): cert {r['cert_share']:.2f} "
              f"change {r['plan_differs']*100:5.1f}% regret {r['regret_mean']*100:5.2f}% "
              f"q90 {r['regret_q90']*100:5.1f}% max {r['regret_max']*100:5.1f}%",
              flush=True)
    with open(OUT / "g1_boundary_v2.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print("saved", OUT / "g1_boundary_v2.json")


if __name__ == "__main__":
    main()
