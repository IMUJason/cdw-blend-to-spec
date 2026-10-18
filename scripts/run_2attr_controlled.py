"""Two-attribute enforcement under the product-basis certification:
controlled-instance table plus a feasibility-boundary sweep.

The literal-enforcement regime is now infeasible at the calibrated second-
attribute level (e~0.43 against product-basis caps kappa'~1-2%); this runner
(1) reproduces the controlled one- vs two-attribute comparison, and
(2) sweeps the feed-side fines share e and the cap relaxation to locate the
boundary where certification becomes reachable.
"""
import json
from pathlib import Path

import numpy as np

from src.models.blend_net import (BlendNetworkSep, DEFAULT_SEPARATION,
                                  build_gate_instance)

OUT = Path("results") / "blend"


def controlled(n_attr):
    inst = build_gate_instance(seed=11, composition="literature",
                               n_attr=n_attr)
    contam = None if n_attr == 1 else inst["c_base"]
    sol = BlendNetworkSep(inst, inst["b_base"], time_limit=300,
                          contam_share=contam).solve()
    return sol


def sweep():
    rows = []
    for e_mult in (1.0, 0.5, 0.25, 0.15):
        for kap_mult in (1.0, 2.0, 4.0):
            inst = build_gate_instance(seed=11, composition="literature",
                                       n_attr=2)
            e = np.clip(inst["c_base"] * e_mult, 0.0, 0.9)
            for d in inst["demands"]:
                d["kappa2"] = float(d["kappa2"] * kap_mult)
            sol = BlendNetworkSep(inst, inst["b_base"], time_limit=120,
                                  contam_share=e).solve()
            util = sol["utilization_rate"] if sol else 0.0
            cost = sol["objective"] if sol else float("nan")
            rows.append({"e_mult": e_mult, "kap_mult": kap_mult,
                         "util": util, "cost": cost})
            print(f"[e x{e_mult:4.2f} | kap' x{kap_mult:3.1f}] util "
                  f"{util*100:5.1f}% cost {cost:,.0f}", flush=True)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    one = controlled(1)
    two = controlled(2)
    tab = {"one_attr": {"cost": one["objective"],
                        "util": one["utilization_rate"],
                        "sep_Mt": sum(v for k, v in one["sep_tonnage"].items()
                                      if k > 0)},
           "two_attr": {"cost": two["objective"],
                        "util": two["utilization_rate"],
                        "sep_Mt": sum(v for k, v in two["sep_tonnage"].items()
                                      if k > 0)},
           "sweep": sweep()}
    with open(OUT / "twoattr_controlled.json", "w") as f:
        json.dump(tab, f, indent=2, default=float)
    print("saved", OUT / "twoattr_controlled.json")


if __name__ == "__main__":
    main()
