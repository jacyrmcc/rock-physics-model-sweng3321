"""ForwardModel (Design 2.4): theoretical rock-physics model (the "RPM").

Workflow (Design 2.4.1):
  1. VRH average of mineral end-members  -> K_s, mu_s, rho_s
  2. Batzle-Wang brine + hydrocarbon, Wood's mixing -> K_fl, rho_fl
  3. Hertz-Mindlin at critical porosity + modified HS bound -> K_dry, mu_dry
  4. Gassmann fluid substitution -> K_sat (mu_sat = mu_dry)
  5. rho_b = (1 - phi) rho_s + phi rho_fl
  6. Vp = sqrt((K + 4/3 mu)/rho), Vs = sqrt(mu/rho)

Units: moduli GPa, density g/cc, pressure MPa, velocity km/s.
(GPa / (g/cc) = (km/s)^2, so no unit conversion is needed for velocities.)

>>> NOTEBOOK HOOK: if your RockPhysics_Theoretical_Model notebook uses a
>>> different frame model or fluid model, replace the body of the matching
>>> private method; keep the signatures so the pipeline is unaffected.
"""
from __future__ import annotations

import numpy as np

from .entities import TARGETS, WellLogDataset

# Reuse from RockPhysics_Theoretical_Model jupyter notebook
LITHOLOGIES = ("VSH", "VSAND", "VCARB")
MINERAL_FOR = {"VSH": "clay", "VSAND": "quartz", "VCARB": "calcite"}


# ----------------------------------------------------------------------------
# Batzle & Wang (1992) fluid properties. T in deg C, P in MPa.
# Returns (K in GPa, rho in g/cc).
# ----------------------------------------------------------------------------
_W = np.array([
    [1402.85, 1.524, 3.437e-3, -1.197e-5],
    [4.871, -0.0111, 1.739e-4, -1.628e-6],
    [-0.04783, 2.747e-4, -2.135e-6, 1.237e-8],
    [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10],
    [-2.197e-7, 7.987e-10, 5.230e-11, -4.614e-13],
])


def brine(salinity: float, T: float, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, float)
    S = salinity
    rho_w = 1 + 1e-6 * (-80 * T - 3.3 * T**2 + 0.00175 * T**3 + 489 * P - 2 * T * P
                        + 0.016 * T**2 * P - 1.3e-5 * T**3 * P - 0.333 * P**2
                        - 0.002 * T * P**2)
    rho_b = rho_w + S * (0.668 + 0.44 * S + 1e-6 * (300 * P - 2400 * P * S + T * (
        80 + 3 * T - 3300 * S - 13 * P + 47 * P * S)))
    v_w = sum(_W[i, j] * T**i * P**j for i in range(5) for j in range(4))
    v_b = (v_w + S * (1170 - 9.6 * T + 0.055 * T**2 - 8.5e-5 * T**3 + 2.6 * P
                      - 0.0029 * T * P - 0.0476 * P**2)
           + S**1.5 * (780 - 10 * P + 0.16 * P**2) - 1820 * S**2)
    k_b = rho_b * (v_b / 1000.0) ** 2  # g/cc * (km/s)^2 = GPa
    return k_b, rho_b


def gas(G: float, T: float, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, float)
    Ta = T + 273.15
    Ppr = P / (4.892 - 0.4048 * G)
    Tpr = Ta / (94.72 + 170.75 * G)
    a = 0.03 + 0.00527 * (3.5 - Tpr) ** 3
    b = 0.642 * Tpr - 0.007 * Tpr**4 - 0.52
    c = 0.109 * (3.85 - Tpr) ** 2
    d = np.exp(-(0.45 + 8 * (0.56 - 1 / Tpr) ** 2) * Ppr**1.2 / Tpr)
    Z = a * Ppr + b + c * d
    rho = 28.8 * G * P / (Z * 8.314 * Ta)  # g/cc
    dZ = a + c * d * (-(0.45 + 8 * (0.56 - 1 / Tpr) ** 2) * 1.2 * Ppr**0.2 / Tpr)
    gamma0 = (0.85 + 5.6 / (Ppr + 2) + 27.1 / (Ppr + 3.5) ** 2
              - 8.7 * np.exp(-0.65 * (Ppr + 1)))
    K = P * gamma0 / (1 - Ppr / Z * dZ) / 1000.0  # MPa -> GPa
    return K, rho


def dead_oil(api: float, T: float, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, float)
    rho0 = 141.5 / (api + 131.5)
    rho_p = rho0 + (0.00277 * P - 1.71e-7 * P**3) * (rho0 - 1.15) ** 2 + 3.49e-4 * P
    rho = rho_p / (0.972 + 3.81e-4 * (T + 17.78) ** 1.175)
    v = (15450 * (77.1 + api) ** -0.5 - 3.7 * T + 4.64 * P
         + 0.0115 * (0.36 * api**0.5 - 1) * T * P)
    return rho * (v / 1000.0) ** 2, rho


class ForwardModel:
    def __init__(self, end_members: dict, hm_params: dict, fluid_params: dict,
                 frame_bound: str = "lower"):
        self.end_members = end_members
        self.hm_params = hm_params
        self.fluid_params = fluid_params
        if frame_bound not in ("lower", "upper"):
            raise ValueError("frame_bound must be 'lower' or 'upper'")
        self.frame_bound = frame_bound

    @classmethod
    def from_config(cls, cfg: dict) -> "ForwardModel":
        return cls(cfg["end_members"], cfg["hm_params"], cfg["fluid_params"],
                   cfg.get("frame_bound", "lower"))

    # ------------------------------------------------------------------ public
    def predict(self, well: WellLogDataset, target_name: str) -> np.ndarray:
        if target_name not in TARGETS:
            raise ValueError(f"target_name must be one of {TARGETS}, got {target_name!r}")
        vp, vs, _ = self.compute(well)
        return vp if target_name == "VP" else vs

    def compute(self, well: WellLogDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (Vp km/s, Vs km/s, RHO g/cc) for every depth sample.

        (Small addition to the Design: exposes density so that
        SyntheticWellGenerator can fill the RHO column, as UC-5 requires.)
        """
        fr = self._fractions(well)
        mins = [self.end_members[MINERAL_FOR[l]] for l in LITHOLOGIES]
        k_s = self._vrh_average(fr, np.array([m["K"] for m in mins]))
        mu_s = self._vrh_average(fr, np.array([m["mu"] for m in mins]))
        rho_s = fr @ np.array([m["rho"] for m in mins])

        phi = well.column("POR")
        k_fl, rho_fl = self._fluid_properties(well.column("SW"), well.column("PD"))
        k_dry, mu_dry = self._frame_moduli(well, k_s, mu_s)
        k_sat = self._gassmann(k_dry, k_s, k_fl, phi)
        mu_sat = mu_dry
        rho_b = (1 - phi) * rho_s + phi * rho_fl

        if not (np.all(np.isfinite(k_sat)) and np.all(np.isfinite(mu_sat))):
            raise RuntimeError(f"Non-finite modulus computed for well {well.name}")
        vp = np.sqrt((k_sat + 4.0 / 3.0 * mu_sat) / rho_b)
        vs = np.sqrt(mu_sat / rho_b)
        return vp, vs, rho_b

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _fractions(well: WellLogDataset) -> np.ndarray:
        fr = np.column_stack([well.column(l) for l in LITHOLOGIES])
        fr = np.clip(fr, 0.0, None)
        total = fr.sum(axis=1, keepdims=True)
        if np.any(total <= 0):
            raise ValueError(f"Well {well.name}: lithology fractions sum to zero")
        return fr / total  # normalise solid fractions to 1

    @staticmethod
    def _vrh_average(fractions: np.ndarray, moduli: np.ndarray) -> np.ndarray:
        voigt = fractions @ moduli
        with np.errstate(divide="ignore"):
            reuss = 1.0 / (fractions @ (1.0 / np.where(moduli > 0, moduli, np.inf)))
        return 0.5 * (voigt + reuss)

    def _fluid_properties(self, sw: np.ndarray, pd: np.ndarray):
        fp = self.fluid_params
        T = fp["temperature_c"]
        p_fluid = pd if fp.get("pore_pressure_mpa") is None else np.full_like(pd, fp["pore_pressure_mpa"])
        k_w, rho_w = brine(fp["salinity"], T, p_fluid)
        if fp.get("hydrocarbon", "gas") == "gas":
            k_hc, rho_hc = gas(fp["gas_gravity"], T, p_fluid)
        else:
            k_hc, rho_hc = dead_oil(fp["api"], T, p_fluid)
        sw = np.clip(sw, 0.0, 1.0)
        k_fl = 1.0 / (sw / k_w + (1 - sw) / k_hc)  # Wood / Reuss
        rho_fl = sw * rho_w + (1 - sw) * rho_hc
        return k_fl, rho_fl

    def _frame_moduli(self, well: WellLogDataset, k_s: np.ndarray, mu_s: np.ndarray):
        fr = self._fractions(well)
        dom = np.array(LITHOLOGIES)[np.argmax(fr, axis=1)]  # dominant lithology
        C = np.array([self.hm_params[d]["C"] for d in dom])
        f = np.array([self.hm_params[d]["gamma"] for d in dom])
        phi_c = np.array([self.hm_params[d]["phi_c"] for d in dom])

        phi = well.column("POR")
        if np.any(phi < 0) or np.any(phi > phi_c):
            bad = int(np.sum((phi < 0) | (phi > phi_c)))
            raise ValueError(f"Well {well.name}: {bad} porosity sample(s) outside [0, phi_c]")

        P = well.column("PD") / 1000.0  # MPa -> GPa
        nu = (3 * k_s - 2 * mu_s) / (2 * (3 * k_s + mu_s))
        base = C**2 * (1 - phi_c) ** 2 * mu_s**2 * P / (np.pi**2 * (1 - nu) ** 2)
        k_hm = (base / 18.0) ** (1 / 3)
        mu_hm = (2 + 3 * f - nu * (1 + 3 * f)) / (5 * (2 - nu)) * (1.5 * base) ** (1 / 3)

        r = phi / phi_c
        if self.frame_bound == "lower":   # soft-sand: HS lower bound, uses mu_HM
            mu_ref, k_ref = mu_hm, k_hm
        else:                             # stiff-sand: HS upper bound, uses mu_s
            mu_ref, k_ref = mu_s, k_s
        z = mu_ref / 6.0 * (9 * k_ref + 8 * mu_ref) / (k_ref + 2 * mu_ref)
        k_dry = 1.0 / (r / (k_hm + 4 / 3 * mu_ref) + (1 - r) / (k_s + 4 / 3 * mu_ref)) - 4 / 3 * mu_ref
        mu_dry = 1.0 / (r / (mu_hm + z) + (1 - r) / (mu_s + z)) - z
        return k_dry, mu_dry

    @staticmethod
    def _gassmann(k_dry, k_s, k_fl, phi):
        num = (1 - k_dry / k_s) ** 2
        den = phi / k_fl + (1 - phi) / k_s - k_dry / k_s**2
        return k_dry + num / den
