from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.optimize import brentq

R_BAR_CM3 = 83.14  # bar·cm^3·mol^-1·K^-1
R_SI = 8.314       # J·mol^-1·K^-1


@dataclass(frozen=True)
class Component:
    """Constantes de un componente puro."""

    name: str
    Tc: float
    Pc: float
    vc: float
    zc: float
    omega: float
    mu: float
    wagner: Tuple[float, float, float, float]
    ps_exp: Tuple[float, float, float, float]
    pure_B2_mode: str = "polar"


COMPONENTS: Tuple[Component, ...] = (
    Component(
        "metanol",
        512.64,
        80.97,
        118.0,
        0.224,
        0.565,
        1.7,
        (-8.63571, 1.17982, -2.479, -1.024),
        (1.0, 1.5, 2.5, 5.0),
        "polar",
    ),
    Component(
        "etanol",
        513.92,
        61.48,
        167.0,
        0.240,
        0.649,
        1.7,
        (-8.68587, 1.17831, -4.8762, 1.588),
        (1.0, 1.5, 2.5, 5.0),
        "polar",
    ),
    Component(
        "agua",
        647.14,
        220.64,
        55.95,
        0.229,
        0.344,
        1.8,
        (-7.77224, 1.45684, -2.71942, -1.41336),
        (1.0, 1.5, 3.0, 6.0),
        "water",
    ),
)

COMPONENT_INDEX = {comp.name: i for i, comp in enumerate(COMPONENTS)}

# Parámetros NRTL transcritos de la hoja.
_AIJ = np.zeros((3, 3), dtype=float)
_BIJ = np.zeros((3, 3), dtype=float)
_ALPHA = np.zeros((3, 3), dtype=float)

# Par 1-2: metanol-etanol
_AIJ[0, 1] = -3.0554
_AIJ[1, 0] = 1.9434
_BIJ[0, 1] = 1328.6516
_BIJ[1, 0] = -877.3991

# Par 1-3: metanol-agua
_AIJ[0, 2] = -2.6311
_AIJ[2, 0] = 4.8683
_BIJ[0, 2] = 838.5936
_BIJ[2, 0] = -1347.527

# Par 2-3: etanol-agua
_AIJ[1, 2] = -0.9852
_AIJ[2, 1] = 3.7555
_BIJ[1, 2] = 302.2365
_BIJ[2, 1] = -676.0314

for i, j in ((0, 1), (0, 2), (1, 2)):
    _ALPHA[i, j] = 0.3
    _ALPHA[j, i] = 0.3


class NRTLVirialVLE:
    """
    Modelo ELV con:
    - coeficientes de actividad NRTL en la fase líquida
    - segundo coeficiente de virial (Tsonopoulos/Prausnitz) en la fase vapor

    Orden de componentes en todos los vectores:
    [metanol, etanol, agua]
    """

    def __init__(self) -> None:
        self.components = COMPONENTS
        self.aij = _AIJ.copy()
        self.bij = _BIJ.copy()
        self.alpha = _ALPHA.copy()
        (
            self.Tcij,
            self.zcij,
            self.vcij,
            self.pcij,
            self.wij,
            self.mij,
        ) = self._pair_pseudocritical()

    @staticmethod
    def _normalize(x: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(x), dtype=float)
        total = arr.sum()
        if total <= 0:
            raise ValueError("La composición debe tener suma positiva.")
        return arr / total

    @staticmethod
    def _scan_bracket(func, low: float, high: float, n: int = 250) -> Tuple[float, float]:
        grid = np.linspace(low, high, n)
        vals = []
        for value in grid:
            try:
                vals.append(float(func(value)))
            except Exception:
                vals.append(np.nan)
        for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if np.isnan(fa) or np.isnan(fb):
                continue
            if fa == 0.0:
                return a, a
            if fa * fb < 0:
                return a, b
        raise ValueError(
            f"No se encontró cambio de signo en el intervalo [{low}, {high}]."
        )

    def _pair_pseudocritical(self):
        Tcij = np.zeros((3, 3), dtype=float)
        zcij = np.zeros((3, 3), dtype=float)
        vcij = np.zeros((3, 3), dtype=float)
        pcij = np.zeros((3, 3), dtype=float)
        wij = np.zeros((3, 3), dtype=float)
        mij = np.zeros((3, 3), dtype=float)

        for i in range(3):
            for j in range(i + 1, 3):
                ci = self.components[i]
                cj = self.components[j]
                Tcij[i, j] = Tcij[j, i] = np.sqrt(ci.Tc * cj.Tc)
                zcij[i, j] = zcij[j, i] = (ci.zc + cj.zc) / 2.0
                vcij[i, j] = vcij[j, i] = ((ci.vc ** (1 / 3) + cj.vc ** (1 / 3)) / 2.0) ** 3
                pcij[i, j] = pcij[j, i] = (
                    R_SI * Tcij[i, j] * zcij[i, j] / (vcij[i, j] * 1.0e-6) / 100000.0
                )
                wij[i, j] = wij[j, i] = (ci.omega + cj.omega) / 2.0
                mij[i, j] = mij[j, i] = (ci.mu + cj.mu) / 2.0
        return Tcij, zcij, vcij, pcij, wij, mij

    def pure_second_virial(self, T: float):
        B = np.zeros(3, dtype=float)
        ps = np.zeros(3, dtype=float)
        phi_sat = np.zeros(3, dtype=float)

        for i, comp in enumerate(self.components):
            Tr = T / comp.Tc
            mr = 100000.0 * comp.mu ** 2 * comp.Pc / comp.Tc ** 2
            B0 = 0.1445 - 0.33 / Tr - 0.1385 / Tr**2 - 0.0121 / Tr**3 - 0.000607 / Tr**8
            B1 = 0.0637 + 0.331 / Tr**2 - 0.423 / Tr**3 - 0.008 / Tr**8
            if comp.pure_B2_mode == "water":
                B2 = 0.0279 / Tr**6 - 0.0229 / Tr**8
            else:
                B2 = 0.0878 / Tr**6 - (0.00908 + 0.0006957 * mr) / Tr**8
            B[i] = R_BAR_CM3 * comp.Tc / comp.Pc * (B0 + comp.omega * B1 + B2)

            tau = 1.0 - Tr
            A, Bc, C, D = comp.wagner
            e1, e2, e3, e4 = comp.ps_exp
            ps[i] = comp.Pc * np.exp((A * tau**e1 + Bc * tau**e2 + C * tau**e3 + D * tau**e4) / Tr)
            phi_sat[i] = np.exp(ps[i] * 100000.0 * B[i] * 1.0e-6 / (R_SI * T))

        phi_liq_pure = ps * phi_sat
        return {
            "B_pure_cm3_mol": B,
            "ps_bar": ps,
            "phi_sat": phi_sat,
            "phi_liq_pure_bar": phi_liq_pure,
        }

    def pair_second_virial(self, T: float) -> np.ndarray:
        Bij = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(i + 1, 3):
                Tr = T / self.Tcij[i, j]
                mr = 100000.0 * self.mij[i, j] ** 2 * self.pcij[i, j] / self.Tcij[i, j] ** 2
                B0 = 0.083 - 0.422 / Tr**1.6
                B1 = 0.139 - 0.172 / Tr**4.2
                B2 = 0.0878 / Tr**6 - (0.00908 + 0.0006957 * mr) / Tr**8
                Bij[i, j] = Bij[j, i] = (
                    (B0 + self.wij[i, j] * B1 + B2) * R_BAR_CM3 * self.Tcij[i, j] / self.pcij[i, j]
                )
        return Bij

    def nrtl_gamma(self, T: float, x: Iterable[float]):
        x = self._normalize(x)
        tau = self.aij + self.bij / T
        G = np.exp(-self.alpha * tau)
        np.fill_diagonal(tau, 0.0)
        np.fill_diagonal(G, 1.0)

        gamma = np.zeros(3, dtype=float)
        for i in range(3):
            term = 0.0
            denom_i = np.sum(x * G[:, i])
            for j in range(3):
                denom_j = np.sum(x * G[:, j])
                numer_j = np.sum(x * tau[:, j] * G[:, j])
                term += x[j] * tau[j, i] * G[j, i] / denom_i
                term += x[j] * G[i, j] / denom_j * (tau[i, j] - numer_j / denom_j)
            gamma[i] = np.exp(term)

        return {
            "x": x,
            "tau": tau,
            "G": G,
            "ln_gamma": np.log(gamma),
            "gamma": gamma,
        }

    def vapor_fugacity_coefficients(self, T: float, P_bar: float, y: Iterable[float]):
        y = self._normalize(y)
        pure = self.pure_second_virial(T)
        B_pure = pure["B_pure_cm3_mol"]
        Bij = self.pair_second_virial(T)

        Bmix = float(np.sum(y * y * B_pure))
        for i in range(3):
            for j in range(i + 1, 3):
                Bmix += 2.0 * y[i] * y[j] * Bij[i, j]

        phi = np.zeros(3, dtype=float)
        for i in range(3):
            cross = 0.0
            for j in range(3):
                cross += y[j] * (B_pure[i] if i == j else Bij[i, j])
            phi[i] = np.exp(P_bar / (R_BAR_CM3 * T) * (2.0 * cross - Bmix))

        return {
            "y": y,
            "Bij_cm3_mol": Bij,
            "Bmix_cm3_mol": Bmix,
            "phi_v": phi,
        }

    def equilibrium_constants(self, T: float, P_bar: float, x: Iterable[float], y: Iterable[float]):
        liquid = self.nrtl_gamma(T, x)
        vapor = self.vapor_fugacity_coefficients(T, P_bar, y)
        pure = self.pure_second_virial(T)
        K = liquid["gamma"] * pure["phi_liq_pure_bar"] / (vapor["phi_v"] * P_bar)

        return {
            "K": K,
            "gamma": liquid["gamma"],
            "ln_gamma": liquid["ln_gamma"],
            "tau": liquid["tau"],
            "G": liquid["G"],
            "phi_liq_pure_bar": pure["phi_liq_pure_bar"],
            "phi_v": vapor["phi_v"],
            "ps_bar": pure["ps_bar"],
            "B_pure_cm3_mol": pure["B_pure_cm3_mol"],
            "Bmix_cm3_mol": vapor["Bmix_cm3_mol"],
            "Bij_cm3_mol": vapor["Bij_cm3_mol"],
        }

    def bubble_state_at_T(
        self,
        T: float,
        P_bar: float,
        x: Iterable[float],
        y0: Iterable[float] | None = None,
        max_iter: int = 200,
        tol: float = 1e-12,
    ):
        x = self._normalize(x)
        if y0 is None:
            pure = self.pure_second_virial(T)
            gamma = self.nrtl_gamma(T, x)["gamma"]
            y = x * gamma * pure["phi_liq_pure_bar"] / P_bar
            y = self._normalize(y)
        else:
            y = self._normalize(y0)

        result = None
        sum_xK = None
        for _ in range(max_iter):
            result = self.equilibrium_constants(T, P_bar, x, y)
            y_raw = x * result["K"]
            sum_xK = float(y_raw.sum())
            y_new = self._normalize(y_raw)
            if np.max(np.abs(y_new - y)) < tol:
                return {
                    "T_K": T,
                    "P_bar": P_bar,
                    "x": x,
                    "y": y_new,
                    "sum_xK": sum_xK,
                    **result,
                }
            y = 0.5 * y + 0.5 * y_new

        raise RuntimeError("La iteración de punto de burbuja no convergió.")

    def dew_state_at_T(
        self,
        T: float,
        P_bar: float,
        y: Iterable[float],
        x0: Iterable[float] | None = None,
        max_iter: int = 200,
        tol: float = 1e-12,
    ):
        y = self._normalize(y)
        if x0 is None:
            x = y.copy()
        else:
            x = self._normalize(x0)

        vapor = self.vapor_fugacity_coefficients(T, P_bar, y)
        pure = self.pure_second_virial(T)

        result = None
        sum_y_over_K = None
        for _ in range(max_iter):
            liquid = self.nrtl_gamma(T, x)
            K = liquid["gamma"] * pure["phi_liq_pure_bar"] / (vapor["phi_v"] * P_bar)
            x_raw = y / K
            sum_y_over_K = float(x_raw.sum())
            x_new = self._normalize(x_raw)
            result = {
                "K": K,
                "gamma": liquid["gamma"],
                "ln_gamma": liquid["ln_gamma"],
                "tau": liquid["tau"],
                "G": liquid["G"],
                "phi_liq_pure_bar": pure["phi_liq_pure_bar"],
                "phi_v": vapor["phi_v"],
                "ps_bar": pure["ps_bar"],
                "B_pure_cm3_mol": pure["B_pure_cm3_mol"],
                "Bmix_cm3_mol": vapor["Bmix_cm3_mol"],
                "Bij_cm3_mol": vapor["Bij_cm3_mol"],
            }
            if np.max(np.abs(x_new - x)) < tol:
                return {
                    "T_K": T,
                    "P_bar": P_bar,
                    "x": x_new,
                    "y": y,
                    "sum_y_over_K": sum_y_over_K,
                    **result,
                }
            x = 0.5 * x + 0.5 * x_new

        raise RuntimeError("La iteración de punto de rocío no convergió.")

    def bubble_temperature(
        self,
        x: Iterable[float],
        P_bar: float,
        T_bounds: Tuple[float, float] = (300.0, 400.0),
    ):
        x = self._normalize(x)

        def residual(T: float) -> float:
            return self.bubble_state_at_T(T, P_bar, x)["sum_xK"] - 1.0

        a, b = self._scan_bracket(residual, T_bounds[0], T_bounds[1])
        T = a if a == b else brentq(residual, a, b, xtol=1e-12, rtol=1e-10)
        return self.bubble_state_at_T(T, P_bar, x)

    def dew_temperature(
        self,
        y: Iterable[float],
        P_bar: float,
        T_bounds: Tuple[float, float] = (300.0, 400.0),
    ):
        y = self._normalize(y)

        def residual(T: float) -> float:
            return self.dew_state_at_T(T, P_bar, y)["sum_y_over_K"] - 1.0

        a, b = self._scan_bracket(residual, T_bounds[0], T_bounds[1])
        T = a if a == b else brentq(residual, a, b, xtol=1e-12, rtol=1e-10)
        return self.dew_state_at_T(T, P_bar, y)

    @staticmethod
    def _rachford_rice(beta: float, z: np.ndarray, K: np.ndarray) -> float:
        return float(np.sum(z * (K - 1.0) / (1.0 + beta * (K - 1.0))))

    def _solve_beta(self, z: np.ndarray, K: np.ndarray) -> float:
        f0 = self._rachford_rice(0.0, z, K)
        f1 = self._rachford_rice(1.0, z, K)
        if f0 < 0.0:
            return 0.0
        if f1 > 0.0:
            return 1.0
        return brentq(lambda beta: self._rachford_rice(beta, z, K), 0.0, 1.0, xtol=1e-13, rtol=1e-11)

    def flash_tp(
        self,
        z: Iterable[float],
        T: float,
        P_bar: float,
        max_iter: int = 200,
        tol: float = 1e-10,
    ):
        z = self._normalize(z)
        gamma = self.nrtl_gamma(T, z)["gamma"]
        pure = self.pure_second_virial(T)
        K = np.maximum(gamma * pure["phi_liq_pure_bar"] / P_bar, 1.0e-14)

        for _ in range(max_iter):
            beta = self._solve_beta(z, K)
            x = z / (1.0 + beta * (K - 1.0))
            y = K * x
            x = self._normalize(x)
            y = self._normalize(y)
            state = self.equilibrium_constants(T, P_bar, x, y)
            K_new = np.maximum(state["K"], 1.0e-14)
            if np.max(np.abs(np.log(K_new / K))) < tol:
                beta = self._solve_beta(z, K_new)
                x = self._normalize(z / (1.0 + beta * (K_new - 1.0)))
                y = self._normalize(K_new * x)
                state = self.equilibrium_constants(T, P_bar, x, y)
                phase = "dos fases"
                if beta <= 1.0e-12:
                    phase = "liquido subenfriado / punto de burbuja"
                elif beta >= 1.0 - 1.0e-12:
                    phase = "vapor sobrecalentado / punto de rocio"
                return {
                    "T_K": T,
                    "P_bar": P_bar,
                    "z": z,
                    "VF": float(beta),
                    "x": x,
                    "y": y,
                    "phase": phase,
                    **state,
                }
            K = np.exp(0.5 * np.log(K) + 0.5 * np.log(K_new))

        raise RuntimeError("El cálculo flash T-P no convergió.")

    def fixed_vf_state(
        self,
        z: Iterable[float],
        T: float,
        P_bar: float,
        VF: float,
        max_iter: int = 200,
        tol: float = 1e-10,
    ):
        z = self._normalize(z)
        gamma = self.nrtl_gamma(T, z)["gamma"]
        pure = self.pure_second_virial(T)
        K = np.maximum(gamma * pure["phi_liq_pure_bar"] / P_bar, 1.0e-14)

        for _ in range(max_iter):
            x_unnorm = z / (1.0 + VF * (K - 1.0))
            y_unnorm = K * x_unnorm
            x = self._normalize(x_unnorm)
            y = self._normalize(y_unnorm)
            state = self.equilibrium_constants(T, P_bar, x, y)
            K_new = np.maximum(state["K"], 1.0e-14)
            if np.max(np.abs(np.log(K_new / K))) < tol:
                rr = self._rachford_rice(VF, z, K_new)
                x_unnorm = z / (1.0 + VF * (K_new - 1.0))
                y_unnorm = K_new * x_unnorm
                x = self._normalize(x_unnorm)
                y = self._normalize(y_unnorm)
                state = self.equilibrium_constants(T, P_bar, x, y)
                return {
                    "T_K": T,
                    "P_bar": P_bar,
                    "z": z,
                    "VF": float(VF),
                    "x": x,
                    "y": y,
                    "rr": float(rr),
                    **state,
                }
            K = np.exp(0.5 * np.log(K) + 0.5 * np.log(K_new))

        raise RuntimeError("El cálculo a V/F fijo no convergió.")

    def temperature_at_vf(
        self,
        z: Iterable[float],
        P_bar: float,
        VF: float,
        T_bounds: Tuple[float, float] = (300.0, 400.0),
    ):
        z = self._normalize(z)

        def residual(T: float) -> float:
            return self.fixed_vf_state(z, T, P_bar, VF)["rr"]

        a, b = self._scan_bracket(residual, T_bounds[0], T_bounds[1])
        T = a if a == b else brentq(residual, a, b, xtol=1e-12, rtol=1e-10)
        return self.fixed_vf_state(z, T, P_bar, VF)

    def pressure_at_vf(
        self,
        z: Iterable[float],
        T: float,
        VF: float,
        P_bounds: Tuple[float, float] = (0.1, 20.0),
    ):
        z = self._normalize(z)

        def residual(P_bar: float) -> float:
            return self.fixed_vf_state(z, T, P_bar, VF)["rr"]

        a, b = self._scan_bracket(residual, P_bounds[0], P_bounds[1])
        P_bar = a if a == b else brentq(residual, a, b, xtol=1e-12, rtol=1e-10)
        return self.fixed_vf_state(z, T, P_bar, VF)

    def binary_phase_diagram(
        self,
        comp_a: str,
        comp_b: str,
        P_bar: float = 1.013,
        points: int = 21,
        T_bounds: Tuple[float, float] = (300.0, 400.0),
    ):
        if comp_a not in COMPONENT_INDEX or comp_b not in COMPONENT_INDEX:
            raise ValueError("Componentes válidos: metanol, etanol, agua.")
        i = COMPONENT_INDEX[comp_a]
        j = COMPONENT_INDEX[comp_b]
        if i == j:
            raise ValueError("Seleccione dos componentes distintos.")

        bubble_curve: List[Dict[str, float]] = []
        dew_curve: List[Dict[str, float]] = []

        for frac in np.linspace(0.0, 1.0, points):
            x = np.zeros(3, dtype=float)
            x[i] = frac
            x[j] = 1.0 - frac
            bubble = self.bubble_temperature(x, P_bar, T_bounds=T_bounds)
            bubble_curve.append(
                {
                    f"x_{comp_a}": float(bubble["x"][i]),
                    f"y_{comp_a}": float(bubble["y"][i]),
                    "T_K": float(bubble["T_K"]),
                }
            )

        for frac in np.linspace(0.0, 1.0, points):
            y = np.zeros(3, dtype=float)
            y[i] = frac
            y[j] = 1.0 - frac
            dew = self.dew_temperature(y, P_bar, T_bounds=T_bounds)
            dew_curve.append(
                {
                    f"x_{comp_a}": float(dew["x"][i]),
                    f"y_{comp_a}": float(dew["y"][i]),
                    "T_K": float(dew["T_K"]),
                }
            )

        return {"bubble": bubble_curve, "dew": dew_curve}


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serializable(v) for v in value]
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ELV con NRTL + segundo coeficiente de virial (metanol/etanol/agua)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bubble = sub.add_parser("bubble", help="Calcula punto de burbuja a presión dada.")
    bubble.add_argument("--x", nargs=3, type=float, required=True, metavar=("x1", "x2", "x3"))
    bubble.add_argument("--P", type=float, required=True, help="Presión en bar.")
    bubble.add_argument("--Tmin", type=float, default=300.0)
    bubble.add_argument("--Tmax", type=float, default=400.0)

    dew = sub.add_parser("dew", help="Calcula punto de rocío a presión dada.")
    dew.add_argument("--y", nargs=3, type=float, required=True, metavar=("y1", "y2", "y3"))
    dew.add_argument("--P", type=float, required=True, help="Presión en bar.")
    dew.add_argument("--Tmin", type=float, default=300.0)
    dew.add_argument("--Tmax", type=float, default=400.0)

    flash = sub.add_parser("flash-tp", help="Flash isotérmico-isobárico.")
    flash.add_argument("--z", nargs=3, type=float, required=True, metavar=("z1", "z2", "z3"))
    flash.add_argument("--T", type=float, required=True, help="Temperatura en K.")
    flash.add_argument("--P", type=float, required=True, help="Presión en bar.")

    tvf = sub.add_parser("flash-vf-T", help="Calcula T a P y V/F fijados.")
    tvf.add_argument("--z", nargs=3, type=float, required=True, metavar=("z1", "z2", "z3"))
    tvf.add_argument("--P", type=float, required=True, help="Presión en bar.")
    tvf.add_argument("--VF", type=float, required=True, help="Fracción vaporizada.")
    tvf.add_argument("--Tmin", type=float, default=300.0)
    tvf.add_argument("--Tmax", type=float, default=400.0)

    pvf = sub.add_parser("flash-vf-P", help="Calcula P a T y V/F fijados.")
    pvf.add_argument("--z", nargs=3, type=float, required=True, metavar=("z1", "z2", "z3"))
    pvf.add_argument("--T", type=float, required=True, help="Temperatura en K.")
    pvf.add_argument("--VF", type=float, required=True, help="Fracción vaporizada.")
    pvf.add_argument("--Pmin", type=float, default=0.1)
    pvf.add_argument("--Pmax", type=float, default=20.0)

    diagram = sub.add_parser("diagram", help="Genera datos T-x-y para un binario.")
    diagram.add_argument("--comp-a", required=True, choices=list(COMPONENT_INDEX))
    diagram.add_argument("--comp-b", required=True, choices=list(COMPONENT_INDEX))
    diagram.add_argument("--P", type=float, default=1.013, help="Presión en bar.")
    diagram.add_argument("--points", type=int, default=21)
    diagram.add_argument("--Tmin", type=float, default=300.0)
    diagram.add_argument("--Tmax", type=float, default=400.0)

    validate = sub.add_parser("validate", help="Reproduce ejemplos de la hoja de cálculo.")
    validate.add_argument(
        "--pretty",
        action="store_true",
        help="Imprime JSON con sangría para facilitar lectura.",
    )

    return parser


def _validation_cases(model: NRTLVirialVLE):
    cases = {
        "burbuja_binaria_metanol_etanol_1p013bar": model.bubble_temperature(
            [0.95, 0.05, 0.0], 1.013, T_bounds=(330.0, 345.0)
        ),
        "burbuja_ternaria_2bar": model.bubble_temperature(
            [0.3, 0.3, 0.4], 2.0, T_bounds=(350.0, 380.0)
        ),
        "vf_0p5_ternaria_2bar": model.temperature_at_vf(
            [0.3, 0.3, 0.4], 2.0, 0.5, T_bounds=(360.0, 375.0)
        ),
        "rocio_ternaria_2bar": model.dew_temperature(
            [0.3, 0.3, 0.4], 2.0, T_bounds=(350.0, 390.0)
        ),
    }
    return cases


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    model = NRTLVirialVLE()

    if args.command == "bubble":
        result = model.bubble_temperature(args.x, args.P, T_bounds=(args.Tmin, args.Tmax))
    elif args.command == "dew":
        result = model.dew_temperature(args.y, args.P, T_bounds=(args.Tmin, args.Tmax))
    elif args.command == "flash-tp":
        result = model.flash_tp(args.z, args.T, args.P)
    elif args.command == "flash-vf-T":
        result = model.temperature_at_vf(args.z, args.P, args.VF, T_bounds=(args.Tmin, args.Tmax))
    elif args.command == "flash-vf-P":
        result = model.pressure_at_vf(args.z, args.T, args.VF, P_bounds=(args.Pmin, args.Pmax))
    elif args.command == "diagram":
        result = model.binary_phase_diagram(
            args.comp_a, args.comp_b, P_bar=args.P, points=args.points, T_bounds=(args.Tmin, args.Tmax)
        )
    elif args.command == "validate":
        result = _validation_cases(model)
        print(json.dumps(_serializable(result), indent=2 if args.pretty else None, ensure_ascii=False))
        return
    else:
        raise ValueError("Comando no reconocido.")

    print(json.dumps(_serializable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
