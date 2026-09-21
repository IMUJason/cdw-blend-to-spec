# Blend-to-specification CDW network design — minimal reproducibility package

Code and stored results for reproducing every number in the paper
"Blend-to-specification network design for construction and demolition waste
recycling under composition uncertainty: location, technology, and source
separation".

## Contents

```
src/models/blend_net.py      model family: single-period, two-stage SAA /
                             chance-constrained, and source-separation MILPs
                             (docplex; optional indicator-constraint variant)
src/models/case_beijing.py   Beijing case instance, calibrated to public
                             statistics (16 districts, 40 candidates, 6 grades)
scripts/                     one runner per result block (see mapping below)
results/                     stored anchor outputs of every runner
```

## Environment

Python 3.10 with `docplex`, `numpy` and a full CPLEX 22.1.x engine
(CPLEX Studio, e.g. 22.1.1; the free community engine's 1000-variable limit
is not sufficient for the city-scale instances). A conda recipe:

```yaml
name: cdw-blend
dependencies:
  - python=3.10
  - pip
  - pip:
      - docplex>=2.25
      - numpy
```

CPLEX Studio's Python API must be installed from the Studio distribution and
matched to the same Python 3.10.

## Reproduce

From the repository root (module form so `src` resolves):

```
python -m scripts.run_g1_sensitivity            # heterogeneity x price grid
python -m scripts.run_g1_literature             # spatial / temporal perturbations
python -m scripts.run_g1_separation_hierarchy   # separation vs information
python -m scripts.run_g2_strengthening          # LP bounds, big-M / pruning
python -m scripts.run_g2c_portfolio             # pinned-portfolio LP bounds
python -m scripts.run_case_beijing              # policy scenario table
python -m scripts.run_case_beijing_2attr        # two-attribute policy table
python -m scripts.run_market_sweep              # frontier figure data
python -m scripts.run_diagnostics               # batch multiplicity,
                                                # pruning counts, root bounds
python -m scripts.run_parameter_robustness      # parameter robustness + GHG
python -m scripts.run_saa_diagnostics           # VSS + out-of-sample
```

All randomness is seed-controlled; runners overwrite the JSON in `results/`.
MILPs solved under a time limit carry solver noise of the order reported in
the paper (objectives within ~0.1%, utilization within ~0.1 pp).

## Result-to-paper mapping

| Paper item | Runner | Stored result |
|---|---|---|
| Boundary table (portfolio change, regret) | run_g1_sensitivity | results/blend/g1_sensitivity.json |
| Spatial vs temporal table | run_g1_literature | results/blend/g1_literature.json |
| Separation-substitution table | run_g1_separation_hierarchy | results/blend/g1_separation_hierarchy.json |
| LP / MILP-gap table | run_g2_strengthening | results/blend/g2_strengthening.json |
| Pinned-portfolio table | run_g2c_portfolio | results/blend/g2c_portfolio.json |
| Beijing policy table | run_case_beijing | results/case_beijing/policy_scenarios.json |
| Two-attribute Beijing table | run_case_beijing_2attr | results/case_beijing/policy_scenarios_2attr.json |
| Frontier figure data | run_market_sweep | results/case_beijing/market_sweep.json |
| Batch multiplicity, pruning counts, root bounds | run_diagnostics | results/diagnostics/checks.json |
| Parameter robustness, GHG | run_parameter_robustness | results/diagnostics/sensitivity_env.json |
| SAA diagnostics (VSS, out-of-sample) | run_saa_diagnostics | results/diagnostics/saa_diagnostics.json |

The case is semi-synthetic; every calibration choice and its provenance are
documented in the paper's supplementary material.
