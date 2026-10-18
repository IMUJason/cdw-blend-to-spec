"""Beijing case under the two-attribute compliance regime.

The second attribute (soil/clay and non-mineral impurities) is capped by the
class limits of GB/T 25177-2010, which are far tighter than the attribute's
share in raw demolition debris. This run tests whether the policy conclusions
survive when the standard is enforced literally.
"""
import json
import time
from pathlib import Path

import numpy as np

from src.models.case_beijing import build_beijing_case
from src.models.blend_net import BlendNetworkSep, DEFAULT_SEPARATION

OUT = Path("results") / "case_beijing"


def evaluate(seed=2024, charge=95.0, strictness=1.0, sep_cost_mult=1.0,
             market_cover=0.85, n_dem=6, two_attr=False):
    inst = build_beijing_case(seed=seed, n_dem=n_dem, demand_cover=market_cover,
                              n_attr=2 if two_attr else 1)
    inst["landfill_fee"] = charge
    if strictness != 1.0:
        for d in inst["demands"]:
            d["kappa"] = float(min(0.95, d["kappa"] * strictness))
    sep = [(sig, lam, cost * sep_cost_mult) for (sig, lam, cost)
           in DEFAULT_SEPARATION]
    t0 = time.time()
    sol = BlendNetworkSep(inst, inst["b_base"], separation=sep,
                          enforce_spec=True,
                          contam_share=inst["c_base"] if two_attr else None
                          ).solve()
    wall = time.time() - t0
    if sol is None:
        return None
    return {"charge": charge, "strictness": strictness,
            "sep_cost_mult": sep_cost_mult, "market_cover": market_cover,
            "two_attr": two_attr,
            "cost_MCNY": sol["objective"] / 1e6,
            "utilization": sol["utilization_rate"],
            "landfilled_Mt": sol["total_landfilled"] / 1e6,
            "landfill_unprocessed_Mt": sol["landfilled_waste"] / 1e6,
            "landfill_sep_loss_Mt": sol["sep_loss"] / 1e6,
            "sep_total_Mt": sum(v for l, v in sol["sep_tonnage"].items() if l > 0),
            "sep_by_level_Mt": {l: v / 1e6 for l, v in sol["sep_tonnage"].items()},
            "facilities_open": sum(1 for v in sol["open"].values() if v),
            "grades_served_Mt": {d: v / 1e6
                                 for d, v in sol["product_d"].items()},
            "wall_s": wall}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    configs = [("base", {}),
               ("market_cover=0.60", {"market_cover": 0.60}),
               ("market_cover=1.00", {"market_cover": 1.00}),
               ("disposal_charge=60", {"charge": 60.0}),
               ("disposal_charge=140", {"charge": 140.0}),
               ("grade_standard=0.7", {"strictness": 0.7}),
               ("grade_standard=1.4", {"strictness": 1.4}),
               ("separation_cost=2.0", {"sep_cost_mult": 2.0})]
    for label, kw in configs:
        r = evaluate(two_attr=True, **kw)
        if r is None:
            print(f"[2attr {label}] infeasible", flush=True)
            continue
        rows.append({"label": label, **r})
        print("[2attr %-20s] cost %6.0f | util %5.1f%% | landfill %5.2f "
              "(unproc %5.2f sep %5.2f) | sep %5.2f | open %2d | served %s"
              % (label, r["cost_MCNY"], 100 * r["utilization"],
                 r["landfilled_Mt"], r["landfill_unprocessed_Mt"],
                 r["landfill_sep_loss_Mt"], r["sep_total_Mt"],
                 r["facilities_open"],
                 [round(x, 2) for x in r["grades_served_Mt"].values()]),
              flush=True)

    with open(OUT / "policy_scenarios_2attr.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nsaved {OUT/'policy_scenarios_2attr.json'}")


if __name__ == "__main__":
    main()
