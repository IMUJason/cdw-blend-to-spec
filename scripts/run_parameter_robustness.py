"""Parameter robustness and environmental accounting on the Beijing case.

Part A: parameter robustness of the single-attribute base case
  (removal efficiency rho of advanced sorting, processing cost, grade prices)
Part B: second-attribute severity (scales the effective soil/clay-impurity
  level, equivalent to making the contaminant harder/easier to remove under
  the equal-efficiency assumption), on the two-attribute regime
Part C: indicative GHG accounting per policy scenario (post-processing;
  factors are mid-range literature values, flagged indicative).

Each run reports total system cost, utilization, landfilled and its
composition, so the numbers line up with the policy table.
"""
import json
import time
from pathlib import Path

import numpy as np

from src.models.case_beijing import build_beijing_case
from src.models.blend_net import BlendNetworkSep, DEFAULT_SEPARATION

OUT = Path("results") / "diagnostics"


def solve_beijing(charge=95.0, strictness=1.0, sep_cost_mult=1.0,
                  market_cover=0.85, rho_adv=None, proc_mult=1.0,
                  price_mult=1.0, n_attr=1, contam_mult=1.0,
                  fixed_charge=None, return_model=False):
    inst = build_beijing_case(seed=2024, n_attr=n_attr,
                              demand_cover=market_cover)
    inst["landfill_fee"] = charge
    if strictness != 1.0:
        for d in inst["demands"]:
            d["kappa"] = float(min(0.95, d["kappa"] * strictness))
    if price_mult != 1.0:
        for d in inst["demands"]:
            d["price"] = float(d["price"] * price_mult)
    if rho_adv is not None:
        inst["techs"][1]["rho"] = float(rho_adv)
    if proc_mult != 1.0:
        for t in inst["techs"]:
            t["proc_cost"] = float(t["proc_cost"] * proc_mult)
    sep = [(sig, lam, c * sep_cost_mult) for (sig, lam, c) in DEFAULT_SEPARATION]
    contam = None
    if n_attr >= 2:
        contam = np.clip(inst["c_base"] * contam_mult, 0.0, 0.95)
    m = BlendNetworkSep(inst, inst["b_base"], separation=sep,
                        contam_share=contam, time_limit=300)
    sol = m.solve()
    if sol is None:
        return None
    rec = {"cost_MCNY": sol["objective"] / 1e6,
           "utilization": sol["utilization_rate"],
           "landfilled_Mt": sol["total_landfilled"] / 1e6,
           "unproc_Mt": sol["landfilled_waste"] / 1e6,
           "seploss_Mt": sol["sep_loss"] / 1e6,
           "open": sum(1 for v in sol["open"].values() if v)}
    if return_model:
        return rec, m, sol, inst
    return rec


def part_a():
    rows = {}
    for tag, kw in [
        ("base", {}),
        ("rho=0.60", {"rho_adv": 0.60}),
        ("rho=0.80", {"rho_adv": 0.80}),
        ("proc x0.5", {"proc_mult": 0.5}),
        ("proc x2.0", {"proc_mult": 2.0}),
        ("price x0.8", {"price_mult": 0.8}),
        ("price x1.2", {"price_mult": 1.2}),
        ("sep cost x0.5", {"sep_cost_mult": 0.5}),
        ("sep cost x2.0", {"sep_cost_mult": 2.0}),
    ]:
        r = solve_beijing(**kw)
        rows[tag] = r
        print(f"[A|{tag}] cost {r['cost_MCNY']:.0f} util "
              f"{r['utilization']*100:.1f}% landfill {r['landfilled_Mt']:.2f} "
              f"open {r['open']}", flush=True)
    return rows


def part_b():
    rows = {}
    for tag, gm in (("2attr base", 1.0), ("2attr c x0.7", 0.7),
                    ("2attr c x1.4", 1.4)):
        r = solve_beijing(n_attr=2, contam_mult=gm)
        rows[tag] = r
        print(f"[B|{tag}] cost {r['cost_MCNY']:.0f} util "
              f"{r['utilization']*100:.1f}% landfill {r['landfilled_Mt']:.2f} "
              f"open {r['open']}", flush=True)
    return rows


# --- Part C: indicative GHG accounting -------------------------------------
# mid-range literature factors (kg CO2e per tonne unless stated), indicative
EF_PROC = 3.5      # recycled-aggregate processing (crushing/screening)
EF_VIRGIN = 6.0    # avoided virgin crushed-rock aggregate production
EF_LANDFILL = 5.0  # landfilling of inert CDW
EF_TKM = 0.11      # road freight, kg CO2e per t-km
DISPOSAL_KM = 20.0 # assumed average haul to disposal (assumption, flagged)


def ghg(rec_model_sol_inst):
    rec, m, sol, inst = rec_model_sol_inst
    g = m.m.solution.get_value
    I = inst
    dist_sf = (I["t_sf"] - 3.0) / 0.55
    dist_fd = (I["t_fd"] - 2.0) / 0.55
    tkm_in = 0.0
    for (s, f, t, l), var in m.a.items():
        lam = m.levels[l][1]
        tkm_in += g(var) * (1 - lam) * dist_sf[s, f]
    tkm_out = 0.0
    for f in range(I["n_fac"]):
        for t in range(I["n_tech"]):
            for d in range(I["n_dem"]):
                tkm_out += g(m.p[f, t, d]) * dist_fd[f, d]
    delivered = sum(sol["product_d"].values())
    landfilled = sol["total_landfilled"]
    tkm_disposal = DISPOSAL_KM * landfilled
    kg = (EF_PROC * delivered + EF_LANDFILL * landfilled
          + EF_TKM * (tkm_in + tkm_out + tkm_disposal)
          - EF_VIRGIN * delivered)
    return {"net_kt": kg / 1e6,
            "kg_per_t_delivered": kg / max(delivered, 1.0),
            "tkm_in_M": tkm_in / 1e6, "tkm_out_M": tkm_out / 1e6,
            "delivered_Mt": delivered / 1e6,
            "landfilled_Mt": landfilled / 1e6}


def part_c():
    rows = {}
    for tag, kw in [("base", {}),
                    ("market 60%", {"market_cover": 0.60}),
                    ("market 100%", {"market_cover": 1.00}),
                    ("charge 60", {"charge": 60.0}),
                    ("charge 140", {"charge": 140.0}),
                    ("std x0.7", {"strictness": 0.7}),
                    ("2attr base", {"n_attr": 2})]:
        got = solve_beijing(return_model=True, **kw)
        rec, m, sol, inst = got
        e = ghg(got)
        e.update({"cost_MCNY": rec["cost_MCNY"],
                  "utilization": rec["utilization"]})
        rows[tag] = e
        print(f"[C|{tag}] net {e['net_kt']:.0f} kt CO2e "
              f"({e['kg_per_t_delivered']:.1f} kg/t delivered) | util "
              f"{e['utilization']*100:.1f}% cost {e['cost_MCNY']:.0f}",
              flush=True)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"A_param": part_a(), "B_2attr": part_b(), "C_ghg": part_c()}
    with open(OUT / "sensitivity_env.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"saved {OUT/'sensitivity_env.json'}")


if __name__ == "__main__":
    main()
