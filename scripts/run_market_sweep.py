"""Market-absorption sweep for the cost-utilization frontier figure.

Sweeps market cover from 0.60 to 1.00 in steps of 0.05 under both the
single-attribute regime and the literal two-attribute regime, reproducing
results/case_beijing/market_sweep.json (18 rows).
"""
import json
from pathlib import Path

import numpy as np

from src.models.case_beijing import build_beijing_case
from src.models.blend_net import BlendNetworkSep, DEFAULT_SEPARATION

OUT = Path("results") / "case_beijing"


def one(cover, n_attr):
    inst = build_beijing_case(seed=2024, demand_cover=cover, n_attr=n_attr)
    contam = None if n_attr == 1 else inst["c_base"]
    sol = BlendNetworkSep(inst, inst["b_base"], separation=DEFAULT_SEPARATION,
                          contam_share=contam, time_limit=300).solve()
    if sol is None:
        return None
    return {"two_attr": n_attr == 2, "market_cover": cover,
            "cost_MCNY": sol["objective"] / 1e6,
            "utilization": sol["utilization_rate"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cover in np.arange(0.60, 1.001, 0.05):
        cover = round(float(cover), 2)
        for n_attr in (1, 2):
            r = one(cover, n_attr)
            if r is not None:
                rows.append(r)
                print(f"[{'2attr' if n_attr == 2 else '1attr'} cover={cover:.2f}] "
                      f"cost {r['cost_MCNY']:.0f} util "
                      f"{r['utilization']*100:.1f}%", flush=True)
    with open(OUT / "market_sweep.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"saved {OUT / 'market_sweep.json'}")


if __name__ == "__main__":
    main()
