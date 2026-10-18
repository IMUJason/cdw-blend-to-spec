"""Model and formulation diagnostics: batch multiplicity, reachability-pruning
counts, and root-bound comparison of big-M vs indicator formulations of the
certification implication.

A. Does the optimal solution actually certify one line for several demand
   points (the model allows it structurally)? Checked on the Beijing base case
   and on the controlled literature instance.
B. How many (line, grade, scenario) pairs does reachability elimination
   remove on the city-scale instance?
C. Root relaxation bound of the two-stage city-scale model under
   (i) big-M, (ii) big-M + pruning, (iii) indicator, (iv) indicator + pruning.
   The big-M numbers must reproduce the reported 18.38% / 4.75% LP gaps.
"""
import contextlib
import io
import json
import re
from pathlib import Path

import numpy as np

from src.models.blend_net import (BlendNetworkSep, BlendNetworkTwoStage,
                                  build_gate_instance, literature_composition)
from src.models.case_beijing import build_beijing_case

OUT = Path("results") / "diagnostics"
BEST_KNOWN = 12994369.71391476   # best incumbent, K=10 baseline 300 s solve


def multi_cert(tag, model, sol):
    """Count lines certified for / shipping to several demand points."""
    g = model.m.solution.get_value
    I = model.inst
    lines_cert, lines_ship = 0, 0
    detail = []
    for f in range(I["n_fac"]):
        for t in range(I["n_tech"]):
            certs = [d for d in range(I["n_dem"]) if sol["batch"][(f, t, d)]]
            ships = [d for d in range(I["n_dem"])
                     if g(model.p[f, t, d]) > 1e-6]
            if len(certs) > 1:
                lines_cert += 1
            if len(ships) > 1:
                lines_ship += 1
                detail.append(((f, t), certs, [round(g(model.p[f, t, d]), 0)
                                               for d in ships]))
    print(f"[{tag}] lines with >1 certification: {lines_cert}; "
          f"lines shipping to >1 demand: {lines_ship}")
    for d in detail[:6]:
        print(f"    line {d[0]}: certs={d[1]} shipped t={d[2]}")
    return {"lines_multi_cert": lines_cert, "lines_multi_ship": lines_ship,
            "examples": [str(x) for x in detail[:6]]}


def part_a():
    res = {}
    inst = build_beijing_case(seed=2024)
    m = BlendNetworkSep(inst, inst["b_base"])
    sol = m.solve()
    res["beijing"] = multi_cert("Beijing base", m, sol)

    inst2 = build_gate_instance(seed=11, composition="literature")
    m2 = BlendNetworkSep(inst2, inst2["b_base"])
    sol2 = m2.solve()
    res["controlled"] = multi_cert("controlled literature", m2, sol2)
    return res


def city_instance_and_scenarios(k=10):
    inst = build_gate_instance(seed=11, composition="literature",
                               n_sites=40, n_fac=20, n_dem=6,
                               demand_cover=0.85)
    rng = np.random.default_rng(2024)
    pool = [literature_composition(seed=int(rng.integers(1, 10**6)),
                                   n_sites=40)["brick_share"]
            for _ in range(25)]
    return inst, pool[:k]


def part_b(inst, scen):
    """Pruned (f,t,d,k) pairs and rows under the reachability rule."""
    n_prune_pairs = 0
    per_k = []
    F, T, D = inst["n_fac"], inst["n_tech"], inst["n_dem"]
    for k, b in enumerate(scen):
        b_min = float(np.min(b))
        cnt = 0
        for t in range(T):
            rho = inst["techs"][t]["rho"]
            for d in range(D):
                if (1 - rho) * b_min > inst["demands"][d]["kappa"]:
                    cnt += F
        per_k.append(cnt)
        n_prune_pairs += cnt
    total = F * T * D * len(scen)
    print(f"[prune] {n_prune_pairs}/{total} (f,t,d,k) certification pairs "
          f"eliminated ({n_prune_pairs/total:.1%}); per-scenario {per_k}")
    return {"pairs": n_prune_pairs, "total": total, "per_scenario": per_k,
            "frac": n_prune_pairs / total}


def lp_bound(inst, scen, tight_m, prune):
    m = BlendNetworkTwoStage(inst, scen, mode="saa", time_limit=600,
                             tight_m=tight_m, prune_batches=prune,
                             relax_integrality=True)
    sol = m.m.solve(log_output=False)
    return float(sol.objective_value)


def mip_root(inst, scen, indicator, prune, seconds=90):
    """Root relaxation objective parsed from the CPLEX MIP log."""
    m = BlendNetworkTwoStage(inst, scen, mode="saa", time_limit=seconds,
                             prune_batches=prune, indicator=indicator)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.m.solve(log_output=True, time_limit=seconds)
    txt = buf.getvalue()
    root = None
    mm = re.search(r"Root relaxation solution time.*?\n", txt)
    if mm:
        nums = re.findall(r"objective\s*[:=]?\s*([0-9][0-9,.\deE+-]*)",
                          mm.group(0), re.I)
        if nums:
            root = float(nums[0].replace(",", ""))
    if root is None:  # fallback: first node-log line's bound column
        for line in txt.splitlines():
            mo = re.search(r"^\s+\d+\+\s+\d+\s+\S+\s+([0-9.eE+-]+)", line)
            if mo:
                root = float(mo.group(1)); break
    best = None
    try:
        best = float(m.m.solution.objective_value)
    except Exception:
        pass
    return root, best, txt[-400:]


def part_c(inst, scen):
    res = {}
    lp_base = lp_bound(inst, scen, tight_m=False, prune=False)
    lp_prune = lp_bound(inst, scen, tight_m=False, prune=True)
    print(f"[LP] as formulated {lp_base:,.0f}  gap {(1-lp_base/BEST_KNOWN):.2%}")
    print(f"[LP] +reachability {lp_prune:,.0f}  gap {(1-lp_prune/BEST_KNOWN):.2%}")
    res["lp_base"], res["lp_prune"] = lp_base, lp_prune

    for ind, pr in ((False, False), (True, False), (True, True)):
        root, best, tail = mip_root(inst, scen, indicator=ind, prune=pr)
        tag = f"indicator={ind},prune={pr}"
        print(f"[MIP root|90s] {tag}: root={root} best={best}")
        res[f"root_{tag}"] = root
        res[f"best_{tag}"] = best
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"A_multicert": part_a()}
    inst, scen = city_instance_and_scenarios(k=10)
    res["B_prune"] = part_b(inst, scen)
    res["C_bounds"] = part_c(inst, scen)
    with open(OUT / "checks.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"saved {OUT/'checks.json'}")


if __name__ == "__main__":
    main()
