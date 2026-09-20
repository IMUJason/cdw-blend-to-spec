"""Beijing CDW recycling case: semi-synthetic instance calibrated to public
statistics; sources and assumptions are documented in the paper's supplementary material.

Real anchors
    total CDW generation 37.72 Mt (Beijing, 2019); downtown 8.99 Mt vs
        suburban/rural 28.73 Mt  (Waste Management & Research 2020)
    demolition waste rate 2.19 t/m^2 of building floor area for China
        (Waste Management 2024)
    composition by structure type and building age (Miatto et al. 2019;
        Cha et al. 2020; Ittyeipe et al. 2023; Zhang et al. 2019)
    disposal-charge anchor 95 CNY/t (landfill-fee study); general construction
        waste management fee 23.61 CNY/t (Wang et al. 2019, JCP)

Synthesized (explicitly flagged in the paper's supplementary material)
    per-district shares inside the downtown / suburban groups (allocated with a
    documented proxy), facility coordinates for synthesized candidates,
    technology / separation cost parameters, recycled-aggregate prices,
    grade thresholds (pending verification against GB/T 25177-2010)

The instance dict has the same schema as build_gate_instance, so every model in
src/models/blend_net.py runs on it unchanged.
"""
from typing import Dict, Optional

import numpy as np

# district name -> (approximate centroid lat, lon, group)
BEIJING_DISTRICTS = {
    "Dongcheng":  (39.928, 116.416, "downtown"),
    "Xicheng":    (39.913, 116.366, "downtown"),
    "Chaoyang":   (39.921, 116.486, "downtown"),
    "Haidian":    (39.959, 116.298, "downtown"),
    "Fengtai":    (39.858, 116.287, "downtown"),
    "Shijingshan": (39.906, 116.223, "downtown"),
    "Tongzhou":   (39.902, 116.658, "suburban"),
    "Shunyi":     (40.130, 116.654, "suburban"),
    "Changping":  (40.220, 116.231, "suburban"),
    "Daxing":     (39.726, 116.338, "suburban"),
    "Fangshan":   (39.749, 116.143, "suburban"),
    "Mentougou":  (39.940, 116.101, "suburban"),
    "Pinggu":     (40.144, 117.121, "suburban"),
    "Huairou":    (40.316, 116.637, "suburban"),
    "Miyun":      (40.377, 116.843, "suburban"),
    "Yanqing":    (40.457, 115.975, "suburban"),
}

# published totals (Mt) for 2019: downtown vs suburban/rural
GROUP_TOTALS_MT = {"downtown": 8.99, "suburban": 28.73}

# stock profiles used for composition: (brick+masonry share mean, within-std)
STOCK_PROFILE = {
    "downtown": (0.52, 0.06),   # older masonry / brick-concrete stock
    "suburban": (0.30, 0.05),   # newer RC-frame dominated stock
}


def _km_xy(lat: float, lon: float, lat0: float = 39.90,
           lon0: float = 116.40) -> tuple:
    """Planar approximation in kilometres around central Beijing."""
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(lat0))
    return ((lon - lon0) * km_per_deg_lon, (lat - lat0) * km_per_deg_lat)


def build_beijing_case(seed: int = 2024, n_dem: int = 6,
                       n_facilities: int = 40, demand_cover: float = 0.85,
                       n_attr: int = 1
                       ) -> Dict:
    """Assemble the Beijing case instance (16 districts as generation sites).

    n_attr=2 adds `c_base`, the district share of soil/clay and non-mineral
    impurities, drawn from a SEPARATE random stream so that everything else in
    the instance is bit-identical to the single-attribute case; the second
    attribute is then capped by the class limits of GB/T 25177-2010.
    """
    rng = np.random.default_rng(seed)
    names = list(BEIJING_DISTRICTS)
    coords_s, groups = [], []
    for nm in names:
        lat, lon, grp = BEIJING_DISTRICTS[nm]
        coords_s.append(_km_xy(lat, lon))
        groups.append(grp)
    coords_s = np.array(coords_s)

    # tonnages: published group totals split within each group with a
    # documented proxy (equal split here; flagged as an assumption)
    Q = np.zeros(len(names))
    for grp in ("downtown", "suburban"):
        idx = [i for i, g in enumerate(groups) if g == grp]
        Q[idx] = GROUP_TOTALS_MT[grp] * 1e6 / len(idx)

    # composition per site: district stock profile + within-profile noise
    b = np.empty(len(names))
    for i, grp in enumerate(groups):
        mu, sd = STOCK_PROFILE[grp]
        b[i] = np.clip(mu + sd * rng.normal(), 0.10, 0.85)

    # second compliance attribute: soil/clay plus non-mineral impurities.
    # Public demolition-waste composition puts soil near 0.36 and metal+wood
    # near 0.07, i.e. about 0.43 combined. Drawn from a separate stream so that
    # enabling it leaves the rest of the instance unchanged.
    c = None
    if n_attr >= 2:
        rng_c = np.random.default_rng(seed + 4242)
        c = np.clip(0.43 + 0.10 * rng_c.normal(0, 1.0, size=len(names)),
                    0.05, 0.75)

    # candidate facilities: existing-style plants near district edges plus
    # synthesized candidates (flagged); placed on a grid over the study area
    xs, ys = coords_s[:, 0], coords_s[:, 1]
    facilities = []
    for f in range(n_facilities):
        if f < 12:     # sited near the twelve largest generation districts
            k = int(np.argsort(-Q)[f])
            facilities.append({"id": f, "coord": coords_s[k].tolist(),
                               "capacity": float(rng.uniform(0.8e6, 1.4e6)),
                               "fixed_cost": float(rng.uniform(2.0e7, 4.5e7))})
        else:          # synthesized candidates spread over the region
            facilities.append({
                "id": f,
                "coord": [float(rng.uniform(xs.min(), xs.max())),
                          float(rng.uniform(ys.min(), ys.max()))],
                "capacity": float(rng.uniform(0.7e6, 1.3e6)),
                "fixed_cost": float(rng.uniform(1.5e7, 3.5e7))})

    # demand: recycled-aggregate users (road / infrastructure works) located
    # around the region; grade ladder with lower grades absorbing more volume
    targets = [(0.05, 52.0, 1.0), (0.15, 36.0, 1.6), (0.30, 24.0, 2.2),
               (0.10, 44.0, 1.3), (0.22, 30.0, 1.9), (0.40, 18.0, 2.5),
               (0.60, 14.0, 2.8)][:n_dem]
    weights = np.array([w for (_k, _p, w) in targets], dtype=float)
    mean_yield = 0.81
    total_demand = demand_cover * float(Q.sum()) * mean_yield
    qty = weights / weights.sum() * total_demand
    coords_d = np.array([_km_xy(39.99, 116.45), _km_xy(39.83, 116.35),
                         _km_xy(40.10, 116.55)])
    demands = [{"id": d, "coord": coords_d[d % 3].tolist(),
                "quantity": float(qty[d]), "kappa": targets[d][0],
                "price": targets[d][1],
                "label": f"grade kappa<={targets[d][0]}",
                "kappa2": 0.014 + 0.02 * targets[d][0]}
               for d in range(n_dem)]

    techs = [
        {"id": 0, "label": "simple crushing", "yield": 0.90, "rho": 0.00,
         "proc_cost": 8.0, "fixed_mult": 1.00},
        {"id": 1, "label": "advanced sorting", "yield": 0.72, "rho": 0.70,
         "proc_cost": 14.0, "fixed_mult": 1.25},
    ]

    def dist(p, q):
        return float(np.hypot(p[0] - q[0], p[1] - q[1]))

    t_sf = np.array([[0.55 * dist(coords_s[s], facilities[f]["coord"]) + 3.0
                      for f in range(n_facilities)] for s in range(len(names))])
    t_fd = np.array([[0.55 * dist(facilities[f]["coord"], coords_d[d % 3]) + 2.0
                      for d in range(n_dem)] for f in range(n_facilities)])

    return {"n_sites": len(names), "n_fac": n_facilities, "n_dem": n_dem,
            "n_tech": 2, "Q": Q, "b_base": b, "c_base": c,
            "facilities": facilities,
            "demands": demands, "techs": techs, "t_sf": t_sf, "t_fd": t_fd,
            "landfill_fee": 95.0,          # CNY/t, public anchor
            "district_names": names, "district_group": groups}
