"""Physics audit for the rebuilt blend-to-specification family.

Checks, on instance parameters and on solved solutions:
  P1 separation feasibility: kept mass w>0 and lost mass sigma*b+gamma*(1-b)
     is a valid split (non-negative brick and non-brick parts) for every
     (site, level) -- structural under (sigma, gamma), verified numerically.
  P2 technology domain: rho_t * b_s <= 1 - eta_t for every technology and the
     feed domain's maximum brick share (residue can carry the removed brick).
  P3 effective post-separation brick share within [0, 1].
  P4 mass balance closure on a solved separation model:
     delivered + rejected + residue + sep_loss + unprocessed == Q.
  P5 certification on the product basis: reported product brick share
     (1-rho)*B / (eta*I) <= kappa for every certified (line, grade).
Exit code 1 on any violation.
"""
import sys

import numpy as np

from src.models.blend_net import (BlendNetworkSep, BlendNetworkTwoStage,
                                  DEFAULT_SEPARATION, build_gate_instance,
                                  literature_composition)
from src.models.case_beijing import build_beijing_case

fails = []


def check(ok, label, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        fails.append(label)


print("=== P1/P2/P3: parameter physics on both instances ===")
for name, inst in [("controlled-literature",
                    build_gate_instance(seed=11, composition="literature")),
                   ("beijing", build_beijing_case(seed=2024))]:
    b = inst["b_base"]
    b_max = float(np.max(b))
    for sig, gam, _c in DEFAULT_SEPARATION:
        for bs in b:
            w = (1 - sig) * bs + (1 - gam) * (1 - bs)
            lost = sig * bs + gam * (1 - bs)
            check(w > 0, f"{name} sep w>0 (sig={sig})",
                  f"b={bs:.3f} w={w:.3f}")
            check(0 <= lost <= 1 and sig * bs <= lost + 1e-12,
                  f"{name} lost-mass split (sig={sig})",
                  f"lost={lost:.3f} >= sig*b={sig*bs:.3f}")
            eff = bs * (1 - sig) / w
            check(0 <= eff <= 1, f"{name} effective share (sig={sig})",
                  f"eff={eff:.3f}")
    for t in inst["techs"]:
        check(t["rho"] * b_max <= 1 - t["yield"] + 1e-12,
              f"{name} tech domain ({t['label']})",
              f"rho*b_max={t['rho']*b_max:.3f} <= 1-eta={1-t['yield']:.3f}")

print("=== P4/P5: mass balance and product-basis certification ===")
inst = build_gate_instance(seed=11, composition="literature")
b = inst["b_base"]
m = BlendNetworkSep(inst, b, time_limit=120)
sol = m.solve()
Q = float(inst["Q"].sum())
check(abs(sol["mass_balance_gap"]) < 1e-4 * Q, "controlled mass balance",
      f"gap={sol['mass_balance_gap']:.6f} on Q={Q:.0f}")
viol = 0.0
for (f, t), deliv in sol["input_ft"].items():
    if deliv <= 1e-9 or not any(sol["batch"][(f, t, d)]
                                for d in range(inst["n_dem"])):
        continue
    eta = inst["techs"][t]["yield"]
    share = (1 - inst["techs"][t]["rho"]) * sol["brick_in_ft"][(f, t)] / (
        eta * deliv)
    for d in range(inst["n_dem"]):
        if sol["batch"][(f, t, d)]:
            viol = max(viol, share - inst["demands"][d]["kappa"])
check(viol <= 1e-6, "product-basis certification holds",
      f"max excess {viol:.2e}")

inst2 = build_beijing_case(seed=2024)
m2 = BlendNetworkSep(inst2, inst2["b_base"], time_limit=300)
sol2 = m2.solve()
Q2 = float(inst2["Q"].sum())
check(abs(sol2["mass_balance_gap"]) < 1e-4 * Q2, "beijing mass balance",
      f"gap={sol2['mass_balance_gap']:.6f} on Q={Q2:.0f}")
print(f"  beijing base: cost {sol2['objective']/1e6:.0f} M CNY, "
      f"util {sol2['utilization_rate']*100:.1f}%, "
      f"landfilled {sol2['total_landfilled']/1e6:.2f} Mt "
      f"(unproc {sol2['landfilled_waste']/1e6:.2f} / sep "
      f"{sol2['sep_loss']/1e6:.2f} / residue {sol2['residue']/1e6:.2f} / "
      f"rej {sol2['rejected_product']/1e6:.2f})")

print("=== P6: two-stage prune domain (no separation) ===")
scen = [literature_composition(seed=s, n_sites=8)["brick_share"]
        for s in (3, 7, 11)]
m3 = BlendNetworkTwoStage(inst, scen, mode="saa", prune_batches=True,
                          time_limit=60)
sol3 = m3.solve()
check(sol3 is not None, "two-stage with product-basis prune solves",
      f"obj {sol3['objective']:.0f}")

print()
if fails:
    print(f"PHYSICS AUDIT FAILED: {len(fails)} violations")
    sys.exit(1)
print("PHYSICS AUDIT PASSED")
