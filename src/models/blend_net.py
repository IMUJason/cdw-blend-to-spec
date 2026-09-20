"""Blend-to-specification CDW recycling network design.

Model family
------------
  literature_composition      site brick shares calibrated to structure type,
                              building age and district stock profile
  build_gate_instance         controlled instance (sites, facilities, grades)
  BlendNetworkModel           single period, batch certification with grade
                              specifications, big-M certification rows
  evaluate_plan               ex-post cost of a committed plan under a realized
                              composition (certification constraints ACTIVE)
  BlendNetworkTwoStage        two-stage stochastic version: SAA or a joint
                              chance constraint on a recycling-service target
  BlendNetworkSep             single period WITH source separation decisions

Decision variables (single period)
    y[f,t] in {0,1}   open facility f with technology t
    w[f,t,d] in {0,1} certify the batch of line (f,t) for demand d
    a[s,f,t] >= 0     waste tonnage from site s processed on line (f,t)
    p[f,t,d] >= 0     certified product shipped from line (f,t) to demand d
    u[s] >= 0         waste landfilled unprocessed
    r[f,t] >= 0       product landfilled (uncertified)

Linear forms
    input_ft    = sum_s a[s,f,t]
    brick_in_ft = sum_s b_s a[s,f,t]
    out_ft      = eta_t input_ft = sum_d p[f,t,d] + r[f,t]

The distinguishing constraint is batch certification: a line can certify grade d
only if its input composition, after processing, meets the grade specification
kappa_d (the maximum admissible brick share). With source separation the input
mix is (1-lambda_l) a delivered tonnage and b_s (1-sigma_l) a contaminant mass.

NOTE: this module was reconstructed after an accidental overwrite; it is
validated against the stored experiment outputs in results/blend/.
"""

from typing import Dict, Optional

import numpy as np

try:
    from docplex.mp.model import Model
except ImportError:  # pragma: no cover
    Model = None


# --------------------------------------------------------------------------
# separation levels
# --------------------------------------------------------------------------

DEFAULT_SEPARATION = [
    # (sigma, lam, cost_per_raw_ton)
    #  sigma = fraction of brick/masonry MASS removed
    #  lam   = fraction of TOTAL mass lost to the removed (contaminant) stream
    #  level 0 (no separation) is always included
    (0.35, 0.18, 6.0),    # screening: removes coarse masonry, modest mass loss
    (0.70, 0.35, 14.0),   # advanced source separation: aggressive but lossy
    (0.85, 0.18, 10.0),   # selective deconstruction: high brick removal and
                          # LOW mass loss (masonry is collected separately,
                          # not discarded); cost reflects the published
                          # 17-25% premium of deconstruction over
                          # mechanical demolition on the demolition contract
]


# --------------------------------------------------------------------------
# composition calibration
# --------------------------------------------------------------------------

def literature_composition(seed: int = 11, n_sites: int = 8,
                           spatial_spread: float = 1.0,
                           temporal_shift: float = 0.0) -> Dict:
    """Per-site brick+masonry share calibrated to published composition ranges.

    Anchors (brick/masonry share of demolition waste, by structure type & age):
      Miatto et al. (2019, Resources, Conservation & Recycling) - the brick
      share of demolition waste in Padua falls from 59% for 1954 buildings to
      41% for 2007 buildings;
      Cha et al. (2020, J. Cleaner Production) - RC versus concrete-brick
      structures differ in demolition-waste recycling potential;
      Ittyeipe et al. (2023) - framed and load-bearing structures differ
      markedly in their concrete/masonry split;
      Hoang et al. (2020, Waste Management) - site-to-site composition variation
      across fifteen demolition sites;
      Zhang et al. (2019, ESPR) - structure-driven differences for China.

    A city's demolition portfolio is a mixture of archetypes, so the site-level
    share inherits the between-archetype spread plus within-archetype noise.
    Sites belong to district stock profiles because urban building stock is
    spatially segregated by era (the basis of age-resolved stock mapping).

    spatial_spread : scales the within-archetype spread.
    temporal_shift : tilts the district mix toward the old core, representing an
        urban-renewal wave: the mean share rises while cross-site dispersion
        falls.
    """
    rng = np.random.default_rng(seed + 7717)
    archetypes = [
        ("RC frame, post-2000", 0.22, 0.05),
        ("RC frame, 1980-2000", 0.32, 0.06),
        ("brick-concrete, 1980-2000", 0.45, 0.06),
        ("brick-concrete, pre-1980", 0.52, 0.06),
        ("load-bearing masonry, pre-1980", 0.58, 0.05),
    ]
    profiles = {
        "old core": np.array([0.03, 0.07, 0.15, 0.35, 0.40]),
        "inner ring": np.array([0.12, 0.18, 0.40, 0.20, 0.10]),
        "new district": np.array([0.45, 0.35, 0.15, 0.04, 0.01]),
    }
    profile_names = list(profiles)
    profile_weights = np.array([0.30, 0.40, 0.30])
    if temporal_shift != 0.0:
        # a renewal wave concentrates demolition on OLDER stock, i.e. tilts the
        # portfolio toward the old core
        profile_weights = np.clip(
            profile_weights + temporal_shift * np.array([0.5, -0.2, -0.3]),
            0.0, None)
        profile_weights = profile_weights / profile_weights.sum()

    b = np.empty(n_sites)
    assigned = []
    for s in range(n_sites):
        pname = profile_names[int(rng.choice(len(profile_names),
                                             p=profile_weights))]
        assigned.append(pname)
        mix = profiles[pname]
        conc = np.maximum(mix, 1e-3) * 25.0
        pi = rng.dirichlet(conc)
        mean_s = float(sum(pi[i] * archetypes[i][1]
                           for i in range(len(archetypes))))
        within = float(sum(pi[i] * archetypes[i][2]
                           for i in range(len(archetypes))))
        b[s] = np.clip(mean_s + spatial_spread * within * rng.normal(0, 1.0),
                       0.05, 0.80)
    return {"brick_share": b, "archetypes": archetypes,
            "profiles": {k: v.tolist() for k, v in profiles.items()},
            "profile_assignment": assigned,
            "mean": float(b.mean()), "std": float(b.std())}


# --------------------------------------------------------------------------
# instance construction
# --------------------------------------------------------------------------

def build_gate_instance(n_sites: int = 8, n_fac: int = 5, seed: int = 11,
                        comp_spread: float = 1.0,
                        composition: Optional[str] = None,
                        n_dem: int = 3,
                        demand_cover: float = 0.85,
                        n_attr: int = 1) -> Dict:
    """Small CDW recycling instance with plausible magnitudes.

    Composition means follow public demolition-waste material shares; the
    "brick share" is the brick+masonry contaminant fraction that governs
    recycled-aggregate quality. composition="literature" replaces the generic
    spread with the archetype-calibrated vector.

    Demand quantities are calibrated so that aggregate demand absorbs
    `demand_cover` of the product the collected waste can yield; otherwise
    forced landfilling swamps any quality-driven decision.

    n_attr=2 additionally supplies `c_base`, the per-site share of soil/clay and
    non-mineral impurities, so the second compliance attribute of the product
    standard can be imposed; its grade caps follow the clay-lump and impurity
    limits of GB/T 25177-2010, derived in the paper.
    """
    rng = np.random.default_rng(seed)
    coords_s = rng.uniform(0, 100, size=(n_sites, 2))
    coords_f = rng.uniform(10, 90, size=(n_fac, 2))
    coords_d = rng.uniform(15, 85, size=(3, 2))

    Q = np.round(rng.uniform(3000, 9000, size=n_sites), 1)
    if composition == "literature":
        b = literature_composition(seed=seed, n_sites=n_sites)["brick_share"]
    else:
        b = np.clip(0.30 + comp_spread * rng.normal(0, 0.10, size=n_sites),
                    0.02, 0.85)
    # second attribute: soil/clay plus non-mineral impurities. Public
    # demolition-waste composition puts soil near 0.36 and metal+wood near
    # 0.07; site-level variation follows the same district logic as masonry.
    # Drawn from a SEPARATE stream so that turning the attribute on leaves
    # every other instance element (tonnages, facilities, demands, masonry
    # vector) bit-identical -- the comparison is then controlled.
    c = None
    if n_attr >= 2:
        rng_c = np.random.default_rng(seed + 4242)
        base_c = 0.43
        c = np.clip(base_c + 0.10 * comp_spread
                    * rng_c.normal(0, 1.0, size=n_sites), 0.05, 0.75)

    facilities = [{
        "id": f, "coord": coords_f[f],
        "capacity": float(rng.uniform(12000, 22000)),
        "fixed_cost": float(rng.uniform(3.0e5, 9.0e5)),
    } for f in range(n_fac)]

    techs = [
        {"id": 0, "label": "simple crushing", "yield": 0.90, "rho": 0.00,
         "proc_cost": 8.0, "fixed_mult": 1.00},
        {"id": 1, "label": "advanced sorting", "yield": 0.72, "rho": 0.70,
         "proc_cost": 14.0, "fixed_mult": 1.25},
    ]

    targets = [(0.05, 52.0, 1.0), (0.15, 36.0, 1.6), (0.30, 24.0, 2.2)]
    if n_dem > 3:
        targets = targets + [(0.10, 44.0, 1.3), (0.22, 30.0, 1.9),
                             (0.40, 18.0, 2.5), (0.60, 14.0, 2.8)][: n_dem - 3]
    weights = np.array([w for (_k, _p, w) in targets], dtype=float)
    mean_yield = 0.81
    total_demand = demand_cover * float(Q.sum()) * mean_yield
    qty = weights / weights.sum() * total_demand
    demands = [{"id": d, "coord": coords_d[d % 3], "quantity": float(qty[d]),
                "kappa": targets[d][0], "price": targets[d][1],
                # second-attribute cap: clay lumps plus non-mineral impurities,
                # from the class limits of GB/T 25177-2010 (0.5/0.7/1.0% clay
                # lumps; 1.0% impurities for the strictest class). The linear
                # rule below reproduces those limits at the class thresholds
                # and interpolates between them.
                "kappa2": 0.014 + 0.02 * targets[d][0],
                "label": f"grade kappa<={targets[d][0]}"}
               for d in range(n_dem)]

    dist = lambda p, q: float(np.hypot(p[0] - q[0], p[1] - q[1]))
    t_sf = np.array([[0.55 * dist(coords_s[s], coords_f[f]) + 3.0
                      for f in range(n_fac)] for s in range(n_sites)])
    t_fd = np.array([[0.55 * dist(coords_f[f], coords_d[d % 3]) + 2.0
                      for d in range(n_dem)] for f in range(n_fac)])

    return {"n_sites": n_sites, "n_fac": n_fac, "n_dem": n_dem, "n_tech": 2,
            "Q": Q, "b_base": b, "c_base": c,
            "facilities": facilities, "demands": demands,
            "techs": techs, "t_sf": t_sf, "t_fd": t_fd, "landfill_fee": 70.0}


# --------------------------------------------------------------------------
# single-period model (no separation)
# --------------------------------------------------------------------------

class BlendNetworkModel:
    """MILP for blend-to-specification CDW network design, no separation.

    brick_share : site brick shares used in THIS solve (one composition
        realization). enforce_spec=False builds the composition-IGNORANT
        variant, used to construct mean-composition plans.
    fix_open / fix_batch : pin a committed plan for ex-post evaluation.
    """

    def __init__(self, inst: Dict, brick_share, enforce_spec: bool = True,
                 fix_open: Optional[Dict] = None,
                 fix_batch: Optional[Dict] = None,
                 contam_share=None):
        self.inst = inst
        self.b = np.asarray(brick_share, dtype=float)
        self.enforce_spec = enforce_spec
        self.fix_open = fix_open
        self.fix_batch = fix_batch
        # Optional SECOND compliance attribute (soil/clay and non-mineral
        # impurities). Passing a vector imposes the standard's second cap
        # demands[d]["kappa2"] alongside the masonry cap; leaving it None
        # builds exactly the single-attribute model.
        self.c = None if contam_share is None else np.asarray(contam_share,
                                                              dtype=float)
        self._build()

    def _build(self):
        I = self.inst
        S, F, D, T = I["n_sites"], I["n_fac"], I["n_dem"], I["n_tech"]
        m = Model(name="blend_net")
        m.parameters.threads.set(1)
        self.m = m

        y = m.binary_var_matrix(F, T, name="y")
        w = m.binary_var_cube(F, T, D, name="w")
        a = m.continuous_var_cube(S, F, T, lb=0.0, name="a")
        p = m.continuous_var_cube(F, T, D, lb=0.0, name="p")
        u = m.continuous_var_list(S, lb=0.0, name="u")
        r = m.continuous_var_matrix(F, T, lb=0.0, name="r")
        self.y, self.w, self.a, self.p, self.u, self.r = y, w, a, p, u, r

        obj = 0.0
        for f in range(F):
            m.add_constraint(m.sum(y[f, t] for t in range(T)) <= 1,
                             ctname=f"one_tech_{f}")
            cap_f = I["facilities"][f]["capacity"]
            for t in range(T):
                obj += (I["facilities"][f]["fixed_cost"]
                        * I["techs"][t]["fixed_mult"] * y[f, t])
                inp = m.sum(a[s, f, t] for s in range(S))
                m.add_constraint(inp <= cap_f * y[f, t], ctname=f"cap_{f}_{t}")
                out = I["techs"][t]["yield"] * inp
                m.add_constraint(out == m.sum(p[f, t, d] for d in range(D))
                                 + r[f, t], ctname=f"out_{f}_{t}")
                brick_in = m.sum(self.b[s] * a[s, f, t] for s in range(S))
                rho_t = I["techs"][t]["rho"]
                for d in range(D):
                    ship_ub = min(I["demands"][d]["quantity"],
                                  I["techs"][t]["yield"] * cap_f)
                    m.add_constraint(p[f, t, d] <= ship_ub * w[f, t, d],
                                     ctname=f"ship_{f}_{t}_{d}")
                    m.add_constraint(w[f, t, d] <= y[f, t],
                                     ctname=f"batch_open_{f}_{t}_{d}")
                    if self.enforce_spec:
                        m.add_constraint(
                            (1 - rho_t) * brick_in
                            <= I["demands"][d]["kappa"] * inp
                            + cap_f * (1 - w[f, t, d]),
                            ctname=f"grade_{f}_{t}_{d}")
                        if self.c is not None:
                            # second attribute: soil/clay and non-mineral
                            # impurities, capped by the standard's class limit
                            m.add_constraint(
                                (1 - rho_t) * m.sum(self.c[s] * a[s, f, t]
                                                    for s in range(S))
                                <= I["demands"][d]["kappa2"] * inp
                                + cap_f * (1 - w[f, t, d]),
                                ctname=f"grade2_{f}_{t}_{d}")
                    obj += (I["t_fd"][f, d]
                            - I["demands"][d]["price"]) * p[f, t, d]
                obj += I["landfill_fee"] * r[f, t]
                for s in range(S):
                    obj += (I["t_sf"][s, f] + I["techs"][t]["proc_cost"]) \
                        * a[s, f, t]
                    m.add_constraint(a[s, f, t] <= I["Q"][s] * y[f, t],
                                     ctname=f"src_{s}_{f}_{t}")

        for s in range(S):
            m.add_constraint(
                m.sum(a[s, f, t] for f in range(F) for t in range(T)) + u[s]
                == I["Q"][s], ctname=f"supply_{s}")
            obj += I["landfill_fee"] * u[s]

        for d in range(D):
            m.add_constraint(
                m.sum(p[f, t, d] for f in range(F) for t in range(T))
                <= I["demands"][d]["quantity"], ctname=f"dem_{d}")

        if self.fix_open is not None:
            for (f, t), v in self.fix_open.items():
                m.add_constraint(y[f, t] == float(v), ctname=f"fy_{f}_{t}")
        if self.fix_batch is not None:
            for (f, t, d), v in self.fix_batch.items():
                m.add_constraint(w[f, t, d] == float(v), ctname=f"fw_{f}_{t}_{d}")

        m.minimize(obj)

    def solve(self, time_limit: Optional[float] = None):
        sol = self.m.solve(log_output=False, time_limit=time_limit or 120)
        if sol is None:
            return None
        I = self.inst
        g = sol.get_value
        out = {"objective": sol.objective_value,
               "open": {(f, t): int(round(g(self.y[f, t]) > 0.5))
                        for f in range(I["n_fac"]) for t in range(I["n_tech"])},
               "batch": {(f, t, d): int(round(g(self.w[f, t, d]) > 0.5))
                         for f in range(I["n_fac"]) for t in range(I["n_tech"])
                         for d in range(I["n_dem"])},
               "input_ft": {}, "brick_in_ft": {}, "product_d": {},
               "landfilled_waste": float(sum(g(self.u[s])
                                             for s in range(I["n_sites"]))),
               "landfilled_product": float(sum(
                   g(self.r[f, t]) for f in range(I["n_fac"])
                   for t in range(I["n_tech"])))}
        for f in range(I["n_fac"]):
            for t in range(I["n_tech"]):
                out["input_ft"][(f, t)] = float(
                    sum(g(self.a[s, f, t]) for s in range(I["n_sites"])))
                out["brick_in_ft"][(f, t)] = float(sum(
                    self.b[s] * g(self.a[s, f, t])
                    for s in range(I["n_sites"])))
        for d in range(I["n_dem"]):
            out["product_d"][d] = float(sum(g(self.p[f, t, d])
                                            for f in range(I["n_fac"])
                                            for t in range(I["n_tech"])))
        return out


def evaluate_plan(inst: Dict, brick_share, plan: Dict,
                  time_limit: float = 120.0) -> Dict:
    """Ex-post cost of a COMMITTED (open, batch) plan under a realized
    composition.

    Plan structure is frozen while sourcing is reoptimized under the realized
    composition with the certification constraints still ACTIVE. A batch whose
    grade cannot be met therefore delivers nothing: its revenue is lost while
    the committed fixed cost is still paid. Because fixing (open, batch) only
    shrinks the composition-aware feasible set, the realized cost upper-bounds
    the aware optimum and the induced regret is non-negative.
    """
    m = BlendNetworkModel(inst, brick_share, enforce_spec=True,
                          fix_open=plan["open"], fix_batch=plan["batch"])
    raw = m.solve(time_limit=time_limit)
    if raw is None:
        return {"feasible": False, "realized_cost": float("inf"),
                "delivered_tonnage": 0.0}
    return {"feasible": True, "realized_cost": raw["objective"],
            "delivered_tonnage": float(sum(raw["product_d"].values())),
            "landfilled_waste": raw["landfilled_waste"],
            "rejected_product": raw["landfilled_product"]}


# --------------------------------------------------------------------------
# two-stage stochastic version: SAA and a joint chance constraint
# --------------------------------------------------------------------------

class BlendNetworkTwoStage:
    """Two-stage stochastic blend-to-specification design.

    First stage : y[f,t] (shared across scenarios)
    Second stage: sourcing, certification, shipping, disposal per scenario

    mode="saa"    minimize fixed cost + mean recourse cost
    mode="chance" additionally require the recycling-service target (landfilled
                  tonnage within `landfill_cap_frac` of collected waste) to hold
                  in at least ceil((1-alpha) K) scenarios

    Flags
      tight_m            use the smallest valid big-M per (line, grade)
      prune_batches      fix w[f,t,d]=0 when (1-rho_t) min_s b_s > kappa_d
      relax_integrality  LP relaxation (used for subtree bounds)
      fix_open           pin the first-stage portfolio
      indicator          model the certification implication with solver-native
                         indicator constraints instead of big-M rows
    """

    def __init__(self, inst: Dict, comp_scenarios, mode: str = "saa",
                 alpha: float = 0.10, landfill_cap_frac: float = 0.40,
                 time_limit: float = 300.0, threads: int = 1,
                 tight_m: bool = False, prune_batches: bool = False,
                 fix_open: Optional[Dict] = None,
                 relax_integrality: bool = False,
                 contam_scenarios=None, indicator: bool = False):
        self.inst = inst
        self.comp = [np.asarray(b, dtype=float) for b in comp_scenarios]
        # optional second attribute per scenario (soil/clay and non-mineral
        # impurities); None reproduces the single-attribute model exactly
        self.comp2 = (None if contam_scenarios is None
                      else [np.asarray(c, dtype=float)
                            for c in contam_scenarios])
        self.K = len(self.comp)
        self.mode = mode
        self.alpha = alpha
        self.landfill_cap_frac = landfill_cap_frac
        self.time_limit = time_limit
        self.threads = threads
        self.tight_m = tight_m
        self.prune_batches = prune_batches
        self.fix_open = fix_open
        self.relax_integrality = relax_integrality
        self.indicator = indicator
        self._build()

    def _build(self):
        I = self.inst
        S, F, D, T, K = (I["n_sites"], I["n_fac"], I["n_dem"], I["n_tech"],
                         self.K)
        m = Model(name=f"blend2s_{self.mode}_K{K}")
        m.parameters.threads.set(self.threads)
        self.m = m

        if self.relax_integrality:
            y = m.continuous_var_matrix(F, T, lb=0.0, ub=1.0, name="y")
        else:
            y = m.binary_var_matrix(F, T, name="y")
        self.y = y
        obj = 0.0
        for f in range(F):
            m.add_constraint(m.sum(y[f, t] for t in range(T)) <= 1,
                             ctname=f"one_tech_{f}")
            for t in range(T):
                obj += (I["facilities"][f]["fixed_cost"]
                        * I["techs"][t]["fixed_mult"] * y[f, t])

        if self.mode == "chance":
            fail = (m.continuous_var_list(K, lb=0.0, ub=1.0, name="fail")
                    if self.relax_integrality
                    else m.binary_var_list(K, name="fail"))
        else:
            fail = None
        self.fail = fail
        total_waste = float(np.sum(I["Q"]))

        self.a, self.w, self.p, self.u, self.r = {}, {}, {}, {}, {}
        for k in range(K):
            b = self.comp[k]
            b_min_k, b_max_k = float(np.min(b)), float(np.max(b))
            c_k = None if self.comp2 is None else self.comp2[k]
            c_min_k = 0.0 if c_k is None else float(np.min(c_k))
            a = m.continuous_var_cube(S, F, T, lb=0.0, name=f"a_{k}")
            if self.relax_integrality:
                w = m.continuous_var_cube(F, T, D, lb=0.0, ub=1.0,
                                          name=f"w_{k}")
            else:
                w = m.binary_var_cube(F, T, D, name=f"w_{k}")
            p = m.continuous_var_cube(F, T, D, lb=0.0, name=f"p_{k}")
            u = m.continuous_var_list(S, lb=0.0, name=f"u_{k}")
            r = m.continuous_var_matrix(F, T, lb=0.0, name=f"r_{k}")
            self.a[k], self.w[k], self.p[k] = a, w, p
            self.u[k], self.r[k] = u, r

            rec = 0.0
            for f in range(F):
                cap_f = I["facilities"][f]["capacity"]
                for t in range(T):
                    inp = m.sum(a[s, f, t] for s in range(S))
                    m.add_constraint(inp <= cap_f * y[f, t],
                                     ctname=f"cap_{k}_{f}_{t}")
                    out = I["techs"][t]["yield"] * inp
                    m.add_constraint(out == m.sum(p[f, t, d]
                                                  for d in range(D)) + r[f, t],
                                     ctname=f"out_{k}_{f}_{t}")
                    brick_in = m.sum(b[s] * a[s, f, t] for s in range(S))
                    contam_in = (None if c_k is None
                                 else m.sum(c_k[s] * a[s, f, t]
                                            for s in range(S)))
                    rho_t = I["techs"][t]["rho"]
                    for d in range(D):
                        kappa_d = I["demands"][d]["kappa"]
                        ship_ub = min(I["demands"][d]["quantity"],
                                      I["techs"][t]["yield"] * cap_f)
                        m.add_constraint(p[f, t, d] <= ship_ub * w[f, t, d],
                                         ctname=f"ship_{k}_{f}_{t}_{d}")
                        m.add_constraint(w[f, t, d] <= y[f, t],
                                         ctname=f"batch_open_{k}_{f}_{t}_{d}")
                        if (self.prune_batches
                                and ((1 - rho_t) * b_min_k > kappa_d
                                     or (self.comp2 is not None
                                         and (1 - rho_t) * c_min_k
                                         > I["demands"][d]["kappa2"]))):
                            m.add_constraint(w[f, t, d] == 0,
                                             ctname=f"prune_{k}_{f}_{t}_{d}")
                            continue
                        big_m = ((1 - rho_t) * b_max_k * cap_f
                                 if self.tight_m else cap_f)
                        if self.indicator:
                            m.add_indicator(
                                w[f, t, d],
                                (1 - rho_t) * brick_in <= kappa_d * inp,
                                active_value=1,
                                name=f"grade_{k}_{f}_{t}_{d}")
                        else:
                            m.add_constraint(
                                (1 - rho_t) * brick_in
                                <= kappa_d * inp + big_m * (1 - w[f, t, d]),
                                ctname=f"grade_{k}_{f}_{t}_{d}")
                        if self.comp2 is not None:
                            if self.indicator:
                                m.add_indicator(
                                    w[f, t, d],
                                    (1 - rho_t) * contam_in
                                    <= I["demands"][d]["kappa2"] * inp,
                                    active_value=1,
                                    name=f"grade2_{k}_{f}_{t}_{d}")
                            else:
                                m.add_constraint(
                                    (1 - rho_t) * contam_in
                                    <= I["demands"][d]["kappa2"] * inp
                                    + big_m * (1 - w[f, t, d]),
                                    ctname=f"grade2_{k}_{f}_{t}_{d}")
                        rec += (I["t_fd"][f, d]
                                - I["demands"][d]["price"]) * p[f, t, d]
                    rec += I["landfill_fee"] * r[f, t]
                    for s in range(S):
                        rec += (I["t_sf"][s, f]
                                + I["techs"][t]["proc_cost"]) * a[s, f, t]
                        m.add_constraint(a[s, f, t] <= I["Q"][s] * y[f, t],
                                         ctname=f"src_{k}_{s}_{f}_{t}")
            for s in range(S):
                m.add_constraint(
                    m.sum(a[s, f, t] for f in range(F) for t in range(T))
                    + u[s] == I["Q"][s], ctname=f"supply_{k}_{s}")
                rec += I["landfill_fee"] * u[s]
            for d in range(D):
                m.add_constraint(
                    m.sum(p[f, t, d] for f in range(F) for t in range(T))
                    <= I["demands"][d]["quantity"], ctname=f"dem_{k}_{d}")
            obj += rec / K

            if self.mode == "chance":
                landfilled = (m.sum(u[s] for s in range(S))
                              + m.sum(r[f, t] for f in range(F)
                                      for t in range(T)))
                cap_k = self.landfill_cap_frac * total_waste
                m.add_constraint(landfilled <= cap_k + total_waste * fail[k],
                                 ctname=f"fail_link_{k}")

        if self.fix_open is not None:
            for (f, t_), v in self.fix_open.items():
                m.add_constraint(y[f, t_] == float(v), ctname=f"fixy_{f}_{t_}")
        if self.mode == "chance":
            import math
            need = int(math.ceil((1.0 - self.alpha) * K))
            m.add_constraint(m.sum(fail[k] for k in range(K)) <= K - need,
                             ctname="chance_count")

        m.minimize(obj)

    def solve(self, log: bool = False):
        sol = self.m.solve(log_output=log, time_limit=self.time_limit)
        if sol is None:
            return None
        I = self.inst
        return {"objective": sol.objective_value,
                "n_vars": self.m.statistics.number_of_variables,
                "n_binaries": self.m.statistics.number_of_binary_variables,
                "n_constraints": self.m.statistics.number_of_constraints,
                "open": {(f, t): int(round(sol.get_value(self.y[f, t]) > 0.5))
                         for f in range(I["n_fac"]) for t in range(I["n_tech"])},
                "solve_time": sol.solve_details.time,
                "gap": sol.solve_details.mip_relative_gap}


# --------------------------------------------------------------------------
# single-period model WITH source separation
# --------------------------------------------------------------------------

class BlendNetworkSep:
    """Blend-to-specification model WITH source separation as a decision.

    For each (site, facility, technology) flow a separation level l is chosen;
    level l removes a fraction sigma_l of the brick mass and loses a fraction
    lam_l of the total mass to a contaminant stream that is landfilled. The
    remaining tonnage (1-lam_l) a is delivered with brick share
    b_s (1-sigma_l) / (1-lam_l). All linear.
    """

    def __init__(self, inst: Dict, brick_share, separation=None,
                 enforce_spec: bool = True, fix_open: Optional[Dict] = None,
                 fix_batch: Optional[Dict] = None, fix_sep: Optional[Dict] = None,
                 time_limit: float = 120.0, contam_share=None):
        self.inst = inst
        self.b = np.asarray(brick_share, dtype=float)
        self.levels = [(0.0, 0.0, 0.0)] + list(
            separation if separation is not None else DEFAULT_SEPARATION)
        self.enforce_spec = enforce_spec
        self.fix_open = fix_open
        self.fix_batch = fix_batch
        self.fix_sep = fix_sep
        self.time_limit = time_limit
        # Optional second compliance attribute (soil/clay and non-mineral
        # impurities); separation is assumed to remove it with the same
        # efficiency as masonry, which is what screening and sorting do.
        self.c = None if contam_share is None else np.asarray(contam_share,
                                                              dtype=float)
        self._build()

    def _build(self):
        I = self.inst
        S, F, D, T = I["n_sites"], I["n_fac"], I["n_dem"], I["n_tech"]
        L = len(self.levels)
        m = Model(name="blend_net_sep")
        m.parameters.threads.set(1)
        self.m = m

        y = m.binary_var_matrix(F, T, name="y")
        w = m.binary_var_cube(F, T, D, name="w")
        a = {(s, f, t, l): m.continuous_var(lb=0.0, name=f"a_{s}_{f}_{t}_{l}")
             for s in range(S) for f in range(F) for t in range(T)
             for l in range(L)}
        p = m.continuous_var_cube(F, T, D, lb=0.0, name="p")
        u = m.continuous_var_list(S, lb=0.0, name="u")
        r = m.continuous_var_matrix(F, T, lb=0.0, name="r")
        self.y, self.w, self.a, self.p, self.u, self.r = y, w, a, p, u, r

        obj = 0.0
        for f in range(F):
            m.add_constraint(m.sum(y[f, t] for t in range(T)) <= 1,
                             ctname=f"one_tech_{f}")
            cap = I["facilities"][f]["capacity"]
            for t in range(T):
                rho_t = I["techs"][t]["rho"]
                obj += (I["facilities"][f]["fixed_cost"]
                        * I["techs"][t]["fixed_mult"] * y[f, t])
                delivered = m.sum((1 - self.levels[l][1]) * a[s, f, t, l]
                                  for s in range(S) for l in range(L))
                brick_in = m.sum(self.b[s] * (1 - self.levels[l][0])
                                 * a[s, f, t, l]
                                 for s in range(S) for l in range(L))
                m.add_constraint(delivered <= cap * y[f, t],
                                 ctname=f"cap_{f}_{t}")
                out = I["techs"][t]["yield"] * delivered
                m.add_constraint(out == m.sum(p[f, t, d] for d in range(D))
                                 + r[f, t], ctname=f"out_{f}_{t}")
                for d in range(D):
                    ship_ub = min(I["demands"][d]["quantity"],
                                  I["techs"][t]["yield"] * cap)
                    m.add_constraint(p[f, t, d] <= ship_ub * w[f, t, d],
                                     ctname=f"ship_{f}_{t}_{d}")
                    m.add_constraint(w[f, t, d] <= y[f, t],
                                     ctname=f"batch_open_{f}_{t}_{d}")
                    if self.enforce_spec:
                        m.add_constraint(
                            (1 - rho_t) * brick_in
                            <= I["demands"][d]["kappa"] * delivered
                            + cap * (1 - w[f, t, d]),
                            ctname=f"grade_{f}_{t}_{d}")
                        if self.c is not None:
                            m.add_constraint(
                                (1 - rho_t) * m.sum(
                                    self.c[s] * (1 - self.levels[l][0])
                                    * a[s, f, t, l]
                                    for s in range(S) for l in range(L))
                                <= I["demands"][d]["kappa2"] * delivered
                                + cap * (1 - w[f, t, d]),
                                ctname=f"grade2_{f}_{t}_{d}")
                    obj += (I["t_fd"][f, d]
                            - I["demands"][d]["price"]) * p[f, t, d]
                obj += I["landfill_fee"] * r[f, t]
                for s in range(S):
                    for l in range(L):
                        sig, lam, cst = self.levels[l]
                        obj += (I["t_sf"][s, f] * (1 - lam)
                                + I["techs"][t]["proc_cost"] * (1 - lam)
                                + cst) * a[s, f, t, l]
                        obj += I["landfill_fee"] * lam * a[s, f, t, l]
                        m.add_constraint(a[s, f, t, l] <= I["Q"][s] * y[f, t],
                                         ctname=f"src_{s}_{f}_{t}_{l}")

        for s in range(S):
            m.add_constraint(
                m.sum(a[s, f, t, l] for f in range(F) for t in range(T)
                      for l in range(L)) + u[s] == I["Q"][s],
                ctname=f"supply_{s}")
            obj += I["landfill_fee"] * u[s]

        for d in range(D):
            m.add_constraint(
                m.sum(p[f, t, d] for f in range(F) for t in range(T))
                <= I["demands"][d]["quantity"], ctname=f"dem_{d}")

        if self.fix_open is not None:
            for (f, t), v in self.fix_open.items():
                m.add_constraint(y[f, t] == float(v), ctname=f"fy_{f}_{t}")
        if self.fix_batch is not None:
            for (f, t, d), v in self.fix_batch.items():
                m.add_constraint(w[f, t, d] == float(v), ctname=f"fw_{f}_{t}_{d}")

        m.minimize(obj)

    def solve(self, time_limit: Optional[float] = None):
        sol = self.m.solve(log_output=False,
                           time_limit=time_limit or self.time_limit)
        if sol is None:
            return None
        I = self.inst
        g = sol.get_value
        out = {"objective": sol.objective_value,
               "open": {(f, t): int(round(g(self.y[f, t]) > 0.5))
                        for f in range(I["n_fac"]) for t in range(I["n_tech"])},
               "batch": {(f, t, d): int(round(g(self.w[f, t, d]) > 0.5))
                         for f in range(I["n_fac"]) for t in range(I["n_tech"])
                         for d in range(I["n_dem"])},
               "product_d": {}, "input_ft": {}, "brick_in_ft": {},
               "sep_tonnage": {}, "landfilled_waste": float(
                   sum(g(self.u[s]) for s in range(I["n_sites"])))}
        for d in range(I["n_dem"]):
            out["product_d"][d] = float(sum(g(self.p[f, t, d])
                                            for f in range(I["n_fac"])
                                            for t in range(I["n_tech"])))
        for f in range(I["n_fac"]):
            for t in range(I["n_tech"]):
                delivered = sum((1 - self.levels[l][1]) * g(self.a[s, f, t, l])
                                for s in range(I["n_sites"])
                                for l in range(len(self.levels)))
                brick = sum(self.b[s] * (1 - self.levels[l][0])
                            * g(self.a[s, f, t, l])
                            for s in range(I["n_sites"])
                            for l in range(len(self.levels)))
                out["input_ft"][(f, t)] = float(delivered)
                out["brick_in_ft"][(f, t)] = float(brick)
        for l, (sig, lam, cst) in enumerate(self.levels):
            out["sep_tonnage"][l] = float(sum(g(self.a[s, f, t, l])
                                              for s in range(I["n_sites"])
                                              for f in range(I["n_fac"])
                                              for t in range(I["n_tech"])))
        # mass accounting: the contaminant stream removed by separation is
        # landfilled too, so it must be counted or utilization is overstated
        delivered = float(sum(out["product_d"].values()))
        sep_loss = float(sum(lam * out["sep_tonnage"][l]
                             for l, (_s, lam, _c) in enumerate(self.levels)))
        rejected = float(sum(g(self.r[f, t]) for f in range(I["n_fac"])
                             for t in range(I["n_tech"])))
        total_waste = float(sum(I["Q"]))
        out["sep_loss"] = sep_loss
        out["rejected_product"] = rejected
        out["total_landfilled"] = out["landfilled_waste"] + sep_loss + rejected
        out["utilization_rate"] = (delivered / total_waste
                                   if total_waste else 0.0)
        return out
