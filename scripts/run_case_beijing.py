"""Beijing case: policy scenarios.

Policy levers swept (all are real instruments for CDW resource utilization):
  disposal_charge  landfill/disposal fee in CNY per tonne (public anchor 95)
  strictness       multiplier on the grade ladder (kappa), i.e. how tight the
                   recycled-aggregate product standards are enforced
  sep_cost_mult    multiplier on source-separation costs (technology subsidy or
                   cost inflation)
  market_cover     how much recycled-aggregate demand the market can absorb
                   relative to achievable product (market-development policy)

Reported per scenario: total system cost, resource-utilization rate, total
landfilled tonnage AND its composition (unprocessed waste / separation losses /
rejected product), separation adoption by level, facilities opened, and which
product grades are served. Composition of landfilling matters: a policy that
lowers landfilling by pushing separation may simply convert unprocessed waste
into separation losses.
"""
import copy
import json
import time
from pathlib import Path

import numpy as np

from src.models.case_beijing import build_beijing_case
from src.models.blend_net import BlendNetworkSep, DEFAULT_SEPARATION

OUT = Path("results") / "case_beijing"


def evaluate(seed=2024, charge=95.0, strictness=1.0, sep_cost_mult=1.0,
             market_cover=0.85, n_dem=6):
    inst = build_beijing_case(seed=seed, n_dem=n_dem, demand_cover=market_cover)
    inst["landfill_fee"] = charge
    # strictness is the multiplier applied to the grade thresholds: values
    # below 1 tighten the product standards (smaller admissible brick share),
    # values above 1 loosen them.
    if strictness != 1.0:
        for d in inst["demands"]:
            d["kappa"] = float(min(0.95, d["kappa"] * strictness))
    sep = [(sig, lam, cost * sep_cost_mult) for (sig, lam, cost)
           in DEFAULT_SEPARATION]
    t0 = time.time()
    sol = BlendNetworkSep(inst, inst["b_base"], separation=sep,
                          enforce_spec=True).solve()
    wall = time.time() - t0
    if sol is None:
        return None
    Q = float(inst["Q"].sum())
    n_open = sum(1 for v in sol["open"].values() if v)
    sep_tons = {l: v / 1e6 for l, v in sol["sep_tonnage"].items()}
    serviced = {d: v / 1e6 for d, v in sol["product_d"].items()}
    return {
        "charge": charge, "strictness": strictness,
        "sep_cost_mult": sep_cost_mult, "market_cover": market_cover,
        "cost_MCNY": sol["objective"] / 1e6,
        "utilization": sol["utilization_rate"],
        "landfilled_Mt": sol["total_landfilled"] / 1e6,
        "landfill_unprocessed_Mt": sol["landfilled_waste"] / 1e6,
        "landfill_sep_loss_Mt": sol["sep_loss"] / 1e6,
        "landfill_rejected_Mt": sol["rejected_product"] / 1e6,
        "sep_total_Mt": sum(v for l, v in sep_tons.items() if l > 0),
        "sep_by_level_Mt": sep_tons,
        "facilities_open": n_open,
        "adv_tech_lines": sum(1 for (f, t), v in sol["open"].items()
                              if v and t == 1),
        "grades_served_Mt": serviced,
        "wall_s": wall,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    base = evaluate()
    rows.append({"label": "base", **base})
    print(f"[base] cost {base['cost_MCNY']:.0f} M CNY | util "
          f"{base['utilization']*100:.1f}% | landfilled {base['landfilled_Mt']:.2f} Mt "
          f"(unproc {base['landfill_unprocessed_Mt']:.2f} / sep-loss "
          f"{base['landfill_sep_loss_Mt']:.2f} / rejected "
          f"{base['landfill_rejected_Mt']:.2f}) | sep {base['sep_total_Mt']:.2f} Mt | "
          f"open {base['facilities_open']}", flush=True)

    sweeps = [
        ("disposal_charge", "charge", [60.0, 140.0]),
        # kappa multiplier: <1 TIGHTENS the product standards, >1 loosens them
        ("grade_standard", "strictness", [0.7, 1.4]),
        ("separation_cost", "sep_cost_mult", [0.5, 2.0]),
        ("market_cover", "market_cover", [0.60, 1.00]),
    ]
    for name, key, vals in sweeps:
        for v in vals:
            kw = dict(charge=95.0, strictness=1.0, sep_cost_mult=1.0,
                      market_cover=0.85)
            kw[key] = v
            r = evaluate(**kw)
            if r is None:
                print(f"[{name}={v}] infeasible", flush=True)
                continue
            rows.append({"label": f"{name}={v}", **r})
            print(f"[{name}={v}] cost {r['cost_MCNY']:.0f} | util "
                  f"{r['utilization']*100:.1f}% | landfilled {r['landfilled_Mt']:.2f} "
                  f"(unproc {r['landfill_unprocessed_Mt']:.2f} / sep "
                  f"{r['landfill_sep_loss_Mt']:.2f} / rej "
                  f"{r['landfill_rejected_Mt']:.2f}) | sep {r['sep_total_Mt']:.2f} | "
                  f"open {r['facilities_open']} | adv {r['adv_tech_lines']} | "
                  f"served {[round(x,2) for x in r['grades_served_Mt'].values()]}",
                  flush=True)

    with open(OUT / "policy_scenarios.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\nsaved {OUT/'policy_scenarios.json'}")


if __name__ == "__main__":
    main()
