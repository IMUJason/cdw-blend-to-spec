"""G1 with source separation, separating COMMITTED from ADAPTIVE decisions.

Decision hierarchy (faithful to practice):
  strategic (committed before composition is known): facility opening + the
      technology installed at each open facility
  operational (adapted after composition is observed): which sites to source,
      how much source separation to buy, which grades to certify

Two commitment assumptions are reported, because both are defensible planning
stances:
  (A) commit open+tech only      -> operational adaptation absorbs composition
  (B) commit open+tech+batches   -> the tactical grade plan is also frozen

The apples-to-apples comparison across model variants runs BOTH through the
same code path: separation=[] is the no-separation model.
"""
import json
import time
from pathlib import Path

import numpy as np

from src.models.blend_net import (build_gate_instance, literature_composition,
                                  BlendNetworkSep, DEFAULT_SEPARATION)

OUT = Path("results") / "blend"


def open_key(sol):
    return tuple(sorted(k for k, v in sol["open"].items() if v))


def batch_key(sol):
    return tuple(sorted(k for k, v in sol["batch"].items() if v))


def ex_post(inst, b, plan, sep, fix_batches, time_limit=120.0):
    m = BlendNetworkSep(inst, b, separation=sep, enforce_spec=True,
                        fix_open=plan["open"],
                        fix_batch=plan["batch"] if fix_batches else None,
                        time_limit=time_limit)
    return m.solve()


def main():
    t0 = time.time()
    inst = build_gate_instance(seed=11, composition="literature")
    n = inst["n_sites"]
    b0 = inst["b_base"].copy()
    rng = np.random.default_rng(11 + 300)

    out = {"instance": {"n_sites": n, "n_fac": inst["n_fac"],
                        "n_dem": inst["n_dem"]},
           "variants": {}}

    for tag, sep in (("no_separation", []),
                     ("with_separation", None)):
        levels = [] if sep == [] else DEFAULT_SEPARATION
        sol_mean = BlendNetworkSep(inst, np.full(n, b0.mean()),
                                   separation=levels, enforce_spec=True).solve()
        ok_mean = open_key(sol_mean)
        bk_mean = batch_key(sol_mean)

        rows = []
        for mode in ("spatial", "renewal"):
            for _ in range(30):
                shift = 0.0 if mode == "spatial" else float(rng.uniform(0.3, 1.0))
                b = literature_composition(seed=int(rng.integers(1, 10**6)),
                                           n_sites=n,
                                           temporal_shift=shift)["brick_share"]
                sol = BlendNetworkSep(inst, b, separation=levels,
                                      enforce_spec=True).solve()
                if sol is None:
                    continue
                rec = {"mode": mode,
                       "open_differs": open_key(sol) != ok_mean,
                       "batch_differs": batch_key(sol) != bk_mean}
                for label, fix_b in (("commit_open", False),
                                     ("commit_all", True)):
                    ev = ex_post(inst, b, sol_mean, levels, fix_b)
                    rec[f"regret_{label}"] = (
                        (ev["objective"] - sol["objective"])
                        / abs(sol["objective"])) if ev else None
                    rec[f"sep_tons_{label}"] = (
                        float(sum(v for l, v in ev["sep_tonnage"].items()
                                  if l > 0)) if ev else None)
                rows.append(rec)
        if not rows:
            continue

        def m_(k):
            v = [r[k] for r in rows if r[k] is not None]
            return float(np.mean(v)) if v else None

        def mx(k):
            v = [r[k] for r in rows if r[k] is not None]
            return float(np.max(v)) if v else None

        var = {
            "n": len(rows),
            "open_differs_rate": m_("open_differs"),
            "batch_differs_rate": m_("batch_differs"),
            "regret_commit_open_mean": m_("regret_commit_open"),
            "regret_commit_open_max": mx("regret_commit_open"),
            "regret_commit_all_mean": m_("regret_commit_all"),
            "regret_commit_all_max": mx("regret_commit_all"),
            "sep_tons_commit_open": m_("sep_tons_commit_open"),
            "sep_tons_commit_all": m_("sep_tons_commit_all"),
            "by_mode": {md: {
                "open_differs_rate": float(np.mean(
                    [r["open_differs"] for r in rows if r["mode"] == md])),
                "regret_commit_open_mean": float(np.mean(
                    [r["regret_commit_open"] for r in rows
                     if r["mode"] == md and r["regret_commit_open"] is not None])),
                "regret_commit_all_mean": float(np.mean(
                    [r["regret_commit_all"] for r in rows
                     if r["mode"] == md and r["regret_commit_all"] is not None])),
                "regret_commit_all_max": float(np.max(
                    [r["regret_commit_all"] for r in rows
                     if r["mode"] == md and r["regret_commit_all"] is not None])),
            } for md in ("spatial", "renewal")},
        }
        out["variants"][tag] = var
        print(f"[{tag}] open-set differs {var['open_differs_rate']*100:.1f}% | "
              f"batch differs {var['batch_differs_rate']*100:.1f}%")
        print(f"    regret  commit open+tech: mean {var['regret_commit_open_mean']*100:.2f}% "
              f"max {var['regret_commit_open_max']*100:.2f}%"
              f"   |   commit all: mean {var['regret_commit_all_mean']*100:.2f}% "
              f"max {var['regret_commit_all_max']*100:.2f}%")
        print(f"    separation bought (commit-open): "
              f"{var['sep_tons_commit_open']:.0f} t", flush=True)

    out["runtime_s"] = time.time() - t0
    with open(OUT / "g1_separation_hierarchy.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT/'g1_separation_hierarchy.json'} in "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
