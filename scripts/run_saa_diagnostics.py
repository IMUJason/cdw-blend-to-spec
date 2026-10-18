"""SAA statistical diagnostics on the city-scale instance.

Reports, for the two-stage blend-to-specification model:
  RP*    expected cost of the SAA first-stage portfolio y*, evaluated
         scenario-by-scenario on an independent out-of-sample set
  EEV*   expected cost of the mean-composition portfolio (the deterministic
         substitute), evaluated on the same out-of-sample set
  VSS    EEV* - RP*  (value of the stochastic solution)
  in-sample SAA objective and K-stability from stored runs.
"""
import json
from pathlib import Path

import numpy as np

from src.models.blend_net import (BlendNetworkModel, BlendNetworkTwoStage,
                                  build_gate_instance, literature_composition)

OUT = Path("results") / "diagnostics"


def city_instance():
    return build_gate_instance(seed=11, composition="literature",
                               n_sites=40, n_fac=20, n_dem=6,
                               demand_cover=0.85)


def scenarios(n, seed0=2024):
    rng = np.random.default_rng(seed0)
    return [literature_composition(seed=int(rng.integers(1, 10**6)),
                                   n_sites=40)["brick_share"]
            for _ in range(n)]


def eval_portfolio(inst, y, b_list, seconds=60):
    """Expected second-stage cost of a fixed first-stage portfolio."""
    tot = 0.0
    for b in b_list:
        m = BlendNetworkModel(inst, b, enforce_spec=True, fix_open=y)
        sol = m.solve(time_limit=seconds)
        tot += sol["objective"]
    return tot / len(b_list)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inst = city_instance()
    train = scenarios(10, seed0=2024)
    test = scenarios(20, seed0=777)          # independent out-of-sample stream

    # RP: SAA portfolio on the training set
    m = BlendNetworkTwoStage(inst, train, mode="saa", time_limit=300,
                             prune_batches=True)
    sol = m.solve()
    y_star = {(f, t): int(round(m.m.solution.get_value(m.y[f, t]) > 0.5))
              for f in range(inst["n_fac"]) for t in range(inst["n_tech"])}
    rp_in = sol["objective"]
    print(f"[RP in-sample  K=10 ] obj {rp_in:,.0f} gap "
          f"{m.m.solve_details.mip_relative_gap:.2%}", flush=True)

    # EEV: portfolio built on the mean composition
    b_mean = np.mean(train, axis=0)
    mm = BlendNetworkModel(inst, b_mean, enforce_spec=True)
    sol_mean = mm.solve(time_limit=120)
    y_mean = sol_mean["open"]
    ee_mean_obj = sol_mean["objective"]
    print(f"[mean-composition model] obj {ee_mean_obj:,.0f}", flush=True)

    # out-of-sample evaluation, in batches to show stability
    res = {"rp_in_sample": rp_in,
           "mean_model_obj": ee_mean_obj,
           "y_star_open": sum(v for v in y_star.values()),
           "y_mean_open": sum(v for v in y_mean.values() if v),
           "blocks": []}
    rp_acc, eev_acc = [], []
    for blk in range(2):
        tb = test[blk * 10:(blk + 1) * 10]
        rp_b = eval_portfolio(inst, y_star, tb)
        eev_b = eval_portfolio(inst, y_mean, tb)
        rp_acc.append(rp_b); eev_acc.append(eev_b)
        res["blocks"].append({"block": blk, "rp": rp_b, "eev": eev_b,
                              "vss": eev_b - rp_b})
        print(f"[oos block {blk}] RP* {rp_b:,.0f}  EEV* {eev_b:,.0f}  "
              f"VSS {eev_b - rp_b:,.0f} ({(eev_b - rp_b)/rp_b:.2%})",
              flush=True)

    rp_oos = float(np.mean(rp_acc)); eev_oos = float(np.mean(eev_acc))
    res.update({"rp_oos": rp_oos, "eev_oos": eev_oos,
                "vss_oos": eev_oos - rp_oos,
                "vss_oos_frac": (eev_oos - rp_oos) / rp_oos})
    print(f"[out-of-sample, 20 scenarios] RP* {rp_oos:,.0f}  EEV* "
          f"{eev_oos:,.0f}  VSS {res['vss_oos']:,.0f} "
          f"({res['vss_oos_frac']:.2%})", flush=True)

    with open(OUT / "saa_diagnostics.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"saved {OUT / 'saa_diagnostics.json'}")


if __name__ == "__main__":
    main()
