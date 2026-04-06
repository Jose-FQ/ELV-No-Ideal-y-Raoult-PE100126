from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import brentq

from nrtl_virial_vle import COMPONENT_INDEX, NRTLVirialVLE

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "experimental_elv_from_sheet.json"
PRECOMPUTED_PATH = APP_DIR / "precomputed_binary_elv_30pts.json"
UNAM_LOGO_PATH = APP_DIR / "Escudo-UNAM-escalable.svg.png"
OWL_LOGO_PATH = APP_DIR / "Logo-Buho.png"



ACKNOWLEDGEMENT = (
    "Agradecimientos al proyecto PAPIME, clave PE100126, titulado: "
    "“Fortalecimiento de la enseñanza experimental del equilibrio químico y físico "
    "utilizando técnicas espectroscópicas” "
)

SYSTEMS: Dict[str, Dict[str, object]] = {
    "agua+etanol": {
        "label": "Agua + etanol",
        "components": ("etanol", "agua"),
        "x_label": "Fracción molar de etanol en líquido, x",
        "y_label": "Fracción molar de etanol en vapor, y",
        "xy_label": "Fracción molar de etanol (x o y)",
        "P_default": 1.013,
        "T_limits": (345.0, 375.0),
    },
    "agua+metanol": {
        "label": "Agua + metanol",
        "components": ("metanol", "agua"),
        "x_label": "Fracción molar de metanol en líquido, x",
        "y_label": "Fracción molar de metanol en vapor, y",
        "xy_label": "Fracción molar de metanol (x o y)",
        "P_default": 1.013,
        "T_limits": (336.0, 374.0),
    },
    "metanol+etanol": {
        "label": "Metanol + etanol",
        "components": ("metanol", "etanol"),
        "x_label": "Fracción molar de metanol en líquido, x",
        "y_label": "Fracción molar de metanol en vapor, y",
        "xy_label": "Fracción molar de metanol (x o y)",
        "P_default": 1.013,
        "T_limits": (337.0, 352.0),
    },
}

MODEL_OPTIONS = {
    "Ideal líquido + ideal vapor (Ley de Raoult)": (True, True),
    "Ideal líquido + vapor no ideal (Virial)": (True, False),
    "Líquido no ideal (NRTL) + vapor ideal": (False, True),
    "Líquido no ideal + vapor no ideal (γ–φ)": (False, False),
}

PLOT_OPTIONS = {
    "y-x a presión fija": "yx",
    "T-x-y a presión fija": "txy",
}

COMPONENTS = ("metanol", "etanol", "agua")


class VLEVariant:
    def __init__(self, liquid_ideal: bool, vapor_ideal: bool) -> None:
        self.base = NRTLVirialVLE()
        self.liquid_ideal = liquid_ideal
        self.vapor_ideal = vapor_ideal

    @staticmethod
    def normalize(x: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(x), dtype=float)
        total = float(arr.sum())
        if total <= 0.0:
            raise ValueError("La suma de composiciones debe ser positiva.")
        return arr / total

    def _gamma(self, T: float, x: Iterable[float]) -> np.ndarray:
        x = self.normalize(x)
        if self.liquid_ideal:
            return np.ones_like(x)
        return self.base.nrtl_gamma(T, x)["gamma"]

    def _pure_terms(self, T: float):
        pure = self.base.pure_second_virial(T)
        fugacity_term = pure["ps_bar"].copy() if self.vapor_ideal else pure["phi_liq_pure_bar"].copy()
        return pure, fugacity_term

    def _phi_vapor(self, T: float, P_bar: float, y: Iterable[float]):
        y = self.normalize(y)
        if self.vapor_ideal:
            return {
                "phi_v": np.ones_like(y),
                "Bmix_cm3_mol": 0.0,
                "Bij_cm3_mol": np.zeros((3, 3), dtype=float),
            }
        return self.base.vapor_fugacity_coefficients(T, P_bar, y)

    def equilibrium_constants(self, T: float, P_bar: float, x: Iterable[float], y: Iterable[float]):
        x = self.normalize(x)
        y = self.normalize(y)
        gamma = self._gamma(T, x)
        pure, fugacity_term = self._pure_terms(T)
        vapor = self._phi_vapor(T, P_bar, y)
        K = gamma * fugacity_term / (vapor["phi_v"] * P_bar)
        return {
            "K": K,
            "gamma": gamma,
            "phi_v": vapor["phi_v"],
            "ps_bar": pure["ps_bar"],
            "f_sat_bar": fugacity_term,
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
        tol: float = 1e-11,
    ):
        x = self.normalize(x)
        if y0 is None:
            pure, fugacity_term = self._pure_terms(T)
            gamma = self._gamma(T, x)
            y = self.normalize(x * gamma * fugacity_term / P_bar)
        else:
            y = self.normalize(y0)

        for _ in range(max_iter):
            result = self.equilibrium_constants(T, P_bar, x, y)
            y_raw = x * result["K"]
            sum_xK = float(y_raw.sum())
            y_new = self.normalize(y_raw)
            if np.max(np.abs(y_new - y)) < tol:
                return {
                    "T_K": float(T),
                    "P_bar": float(P_bar),
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
        tol: float = 1e-11,
    ):
        y = self.normalize(y)
        x = y.copy() if x0 is None else self.normalize(x0)
        pure, fugacity_term = self._pure_terms(T)
        vapor = self._phi_vapor(T, P_bar, y)

        for _ in range(max_iter):
            gamma = self._gamma(T, x)
            K = gamma * fugacity_term / (vapor["phi_v"] * P_bar)
            x_raw = y / K
            sum_y_over_K = float(x_raw.sum())
            x_new = self.normalize(x_raw)
            if np.max(np.abs(x_new - x)) < tol:
                return {
                    "T_K": float(T),
                    "P_bar": float(P_bar),
                    "x": x_new,
                    "y": y,
                    "sum_y_over_K": sum_y_over_K,
                    "K": K,
                    "gamma": gamma,
                    "phi_v": vapor["phi_v"],
                }
            x = 0.5 * x + 0.5 * x_new

        raise RuntimeError("La iteración de punto de rocío no convergió.")

    def _find_bracket(self, residual_func, bounds: Tuple[float, float], guess: float | None = None):
        low, high = bounds

        if guess is not None:
            for span, samples in ((2.0, 12), (4.0, 14), (7.0, 16), (12.0, 18)):
                a = max(low, guess - span)
                b = min(high, guess + span)
                grid = np.linspace(a, b, samples)
                values = []
                for T in grid:
                    try:
                        values.append(float(residual_func(float(T))))
                    except Exception:
                        values.append(np.nan)
                for a1, b1, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
                    if np.isnan(fa) or np.isnan(fb):
                        continue
                    if fa == 0.0:
                        return float(a1), float(a1)
                    if fa * fb < 0.0:
                        return float(a1), float(b1)

        grid = np.linspace(low, high, 50)
        values = []
        for T in grid:
            try:
                values.append(float(residual_func(float(T))))
            except Exception:
                values.append(np.nan)
        for a1, b1, fa, fb in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
            if np.isnan(fa) or np.isnan(fb):
                continue
            if fa == 0.0:
                return float(a1), float(a1)
            if fa * fb < 0.0:
                return float(a1), float(b1)
        raise ValueError(f"No se encontró un intervalo de raíces en {bounds}.")

    def bubble_temperature(
        self,
        x: Iterable[float],
        P_bar: float,
        T_bounds: Tuple[float, float] = (300.0, 400.0),
        T_guess: float | None = None,
    ):
        x = self.normalize(x)

        def residual(T: float) -> float:
            return self.bubble_state_at_T(T, P_bar, x)["sum_xK"] - 1.0

        a, b = self._find_bracket(residual, T_bounds, guess=T_guess)
        T = a if a == b else brentq(residual, a, b, xtol=1e-10, rtol=1e-9, maxiter=80)
        return self.bubble_state_at_T(T, P_bar, x)

    def dew_temperature(
        self,
        y: Iterable[float],
        P_bar: float,
        T_bounds: Tuple[float, float] = (300.0, 400.0),
        T_guess: float | None = None,
    ):
        y = self.normalize(y)

        def residual(T: float) -> float:
            return self.dew_state_at_T(T, P_bar, y)["sum_y_over_K"] - 1.0

        a, b = self._find_bracket(residual, T_bounds, guess=T_guess)
        T = a if a == b else brentq(residual, a, b, xtol=1e-10, rtol=1e-9, maxiter=80)
        return self.dew_state_at_T(T, P_bar, y)


@st.cache_data(show_spinner=False)
def load_experimental_data() -> dict:
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_precomputed_data() -> dict:
    with open(PRECOMPUTED_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def dataframe_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def figure_download_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def json_download_bytes(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return value


def composition_dataframe(composition: Iterable[float], title: str) -> pd.DataFrame:
    composition = np.asarray(list(composition), dtype=float)
    return pd.DataFrame(
        {
            "Conjunto": [title] * 3,
            "Componente": list(COMPONENTS),
            "Fracción molar": composition,
        }
    )


def build_yx_figure(system_key: str, model_df: pd.DataFrame, exp_series: list):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(model_df["x"], model_df["y"], label="Modelo precalculado")
    ax.plot([0, 1], [0, 1], linestyle="--", label="y = x")
    for series in exp_series:
        ax.scatter(series["x"], series["y"], label=series["label"])
    ax.set_xlabel(SYSTEMS[system_key]["x_label"])
    ax.set_ylabel(SYSTEMS[system_key]["y_label"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def build_txy_figure(system_key: str, bubble_df: pd.DataFrame, dew_df: pd.DataFrame, exp_series: list):
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ax.plot(bubble_df["x"], bubble_df["T_K"], label="Modelo burbuja: T-x")
    ax.plot(dew_df["y"], dew_df["T_K"], label="Modelo rocío: T-y")
    for series in exp_series:
        ax.scatter(series["x"], series["T_K"], label=f"{series['label']} T-x")
        ax.scatter(series["y"], series["T_K"], label=f"{series['label']} T-y", marker="s")
    ax.set_xlabel(SYSTEMS[system_key]["xy_label"])
    ax.set_ylabel("Temperatura / K")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def render_header() -> None:
    col1, col2, col3 = st.columns([1.15, 2.8, 0.95])
    with col1:
        if UNAM_LOGO_PATH.exists():
            st.image(str(UNAM_LOGO_PATH), use_container_width=True)
    with col2:
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:0.4rem;">
                <h1 style="margin-bottom:0.1rem;">Equilibrio Líquido-Vapor</h1>
                <h3 style="margin-top:0.1rem; margin-bottom:0.1rem; font-weight:500;">Laboratorio de Equilibrio y Cinética</h3>
                <h4 style="margin-top:0.1rem; margin-bottom:0.6rem; font-weight:400;">2026</h4>
                <div style="font-size:0.92rem; line-height:1.35; max-width:820px; margin:0 auto;">
                    {ACKNOWLEDGEMENT}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        if OWL_LOGO_PATH.exists():
            st.image(str(OWL_LOGO_PATH), use_container_width=True)
    st.divider()


def render_binary_tab(precomputed_data: dict, experimental_data: dict) -> None:
    st.markdown(
        "Esta pestaña usa una **base de datos precalculada** con **30 puntos fijos** para cada sistema y modelo. "
        "La presión se mantiene en **1.013 bar** para que la comparación experimental sea directa."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        system_key = st.selectbox(
            "Sistema binario",
            options=list(SYSTEMS.keys()),
            format_func=lambda key: SYSTEMS[key]["label"],
            key="binary_system",
        )
    with col2:
        plot_key = st.radio(
            "Tipo de gráfica",
            options=list(PLOT_OPTIONS.keys()),
            horizontal=False,
            key="binary_plot",
        )
    with col3:
        model_option = st.selectbox(
            "Modelo termodinámico",
            options=list(MODEL_OPTIONS.keys()),
            index=3,
            key="binary_model",
        )

    pressure_bar = float(SYSTEMS[system_key]["P_default"])
    points = int(precomputed_data["metadata"]["points"])
    exp_series = experimental_data[system_key]["experimental_series"]
    st.caption(f"Sistema: {SYSTEMS[system_key]['label']} | Presión fija: {pressure_bar:.3f} bar | Puntos precalculados: {points}")

    if PLOT_OPTIONS[plot_key] == "yx":
        model_df = pd.DataFrame(precomputed_data[system_key][model_option]["yx"])
        fig = build_yx_figure(system_key, model_df, exp_series)
        st.pyplot(fig)

        m1, m2, m3 = st.columns(3)
        m1.metric("Presión", f"{pressure_bar:.3f} bar")
        m2.metric("Modelo", model_option)
        m3.metric("Puntos", str(points))

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "Descargar curva y-x (CSV)",
                dataframe_download_bytes(model_df[["x", "y", "T_K", "P_bar"]]),
                file_name=f"yx_{system_key}_30pts.csv",
                mime="text/csv",
            )
        with col_b:
            st.download_button(
                "Descargar figura (PNG)",
                figure_download_bytes(fig),
                file_name=f"yx_{system_key}.png",
                mime="image/png",
            )

        st.markdown("**Curva precalculada**")
        st.dataframe(model_df[["x", "y", "T_K", "P_bar"]], use_container_width=True)

    else:
        bubble_df = pd.DataFrame(precomputed_data[system_key][model_option]["txy"]["bubble"])
        dew_df = pd.DataFrame(precomputed_data[system_key][model_option]["txy"]["dew"])
        fig = build_txy_figure(system_key, bubble_df, dew_df, exp_series)
        st.pyplot(fig)

        m1, m2, m3 = st.columns(3)
        m1.metric("Presión", f"{pressure_bar:.3f} bar")
        m2.metric("Modelo", model_option)
        m3.metric("Puntos", str(points))

        combined_df = pd.DataFrame(
            {
                "x_bubble": bubble_df["x"],
                "y_bubble": bubble_df["y"],
                "T_bubble_K": bubble_df["T_K"],
                "x_dew": dew_df["x"],
                "y_dew": dew_df["y"],
                "T_dew_K": dew_df["T_K"],
            }
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "Descargar curvas T-x-y (CSV)",
                dataframe_download_bytes(combined_df),
                file_name=f"txy_{system_key}_30pts.csv",
                mime="text/csv",
            )
        with col_b:
            st.download_button(
                "Descargar figura (PNG)",
                figure_download_bytes(fig),
                file_name=f"txy_{system_key}.png",
                mime="image/png",
            )

        left, right = st.columns(2)
        with left:
            st.markdown("**Curva de burbuja precalculada**")
            st.dataframe(bubble_df[["x", "y", "T_K", "P_bar"]], use_container_width=True)
        with right:
            st.markdown("**Curva de rocío precalculada**")
            st.dataframe(dew_df[["x", "y", "T_K", "P_bar"]], use_container_width=True)

    with st.expander("Datos experimentales de la hoja"):
        for series in exp_series:
            exp_df = pd.DataFrame({"x": series["x"], "y": series["y"], "T_K": series["T_K"]})
            st.markdown(f"**{series['label']}**")
            st.dataframe(exp_df, use_container_width=True)

    with st.expander("Qué representa cada modelo"):
        st.write(
            "- **Ideal líquido + ideal vapor:** Ley de Raoult clásica, con γ = 1 y Φ = 1.\n"
            "- **Ideal líquido + vapor no ideal:** γ = 1 y Φ calculado con segundo coeficiente de virial.\n"
            "- **Líquido no ideal + vapor ideal:** γ con NRTL y Φ = 1.\n"
            "- **Líquido no ideal + vapor no ideal (γ–φ):** γ con NRTL y Φ con segundo coeficiente de virial."
        )


def render_ternary_tab() -> None:
    st.markdown(
        "Ingrese las composiciones de la mezcla ternaria **metanol + etanol + agua** para determinar "
        "el **punto de burbuja** y el **punto de rocío** a presión constante."
    )

    top1, top2, top3 = st.columns([1.7, 1.0, 1.2])
    with top1:
        model_option = st.selectbox(
            "Modelo termodinámico",
            options=list(MODEL_OPTIONS.keys()),
            index=3,
            key="ternary_model",
        )
    with top2:
        pressure_bar = st.number_input("Presión / bar", min_value=0.05, max_value=10.0, value=1.013, step=0.05)
    with top3:
        st.markdown("**Orden de componentes**")
        st.caption("metanol, etanol, agua")

    with st.expander("Opciones numéricas del intervalo de búsqueda"):
        tmin, tmax = st.slider(
            "Intervalo de temperatura / K",
            min_value=300.0,
            max_value=430.0,
            value=(330.0, 390.0),
            step=1.0,
        )

    liquid_ideal, vapor_ideal = MODEL_OPTIONS[model_option]
    model = VLEVariant(liquid_ideal=liquid_ideal, vapor_ideal=vapor_ideal)

    col_bubble, col_dew = st.columns(2)

    with col_bubble:
        st.subheader("Punto de burbuja")
        with st.form("bubble_form"):
            x_metanol = st.number_input("x_metanol", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
            x_etanol = st.number_input("x_etanol", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
            x_agua = st.number_input("x_agua", min_value=0.0, max_value=1.0, value=0.34, step=0.01)
            bubble_submit = st.form_submit_button("Calcular punto de burbuja")

        if bubble_submit:
            try:
                x_input = VLEVariant.normalize([x_metanol, x_etanol, x_agua])
                result = model.bubble_temperature(x_input, float(pressure_bar), T_bounds=(float(tmin), float(tmax)))
                st.success("Cálculo completado.")
                m1, m2 = st.columns(2)
                m1.metric("T de burbuja", f"{result['T_K']:.3f} K")
                m2.metric("T de burbuja", f"{result['T_K'] - 273.15:.3f} °C")
                st.markdown("**Composición líquida normalizada (entrada)**")
                st.dataframe(composition_dataframe(result["x"], "x"), use_container_width=True)
                st.markdown("**Composición de vapor en equilibrio (salida)**")
                st.dataframe(composition_dataframe(result["y"], "y"), use_container_width=True)
                st.markdown("**Constantes de equilibrio K**")
                k_df = pd.DataFrame({"Componente": list(COMPONENTS), "K": result["K"]})
                st.dataframe(k_df, use_container_width=True)
                st.download_button(
                    "Descargar resultado de burbuja (JSON)",
                    json_download_bytes(serializable(result)),
                    file_name="punto_burbuja_ternario.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.error(f"No fue posible calcular el punto de burbuja: {exc}")

    with col_dew:
        st.subheader("Punto de rocío")
        with st.form("dew_form"):
            y_metanol = st.number_input("y_metanol", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
            y_etanol = st.number_input("y_etanol", min_value=0.0, max_value=1.0, value=0.33, step=0.01)
            y_agua = st.number_input("y_agua", min_value=0.0, max_value=1.0, value=0.34, step=0.01)
            dew_submit = st.form_submit_button("Calcular punto de rocío")

        if dew_submit:
            try:
                y_input = VLEVariant.normalize([y_metanol, y_etanol, y_agua])
                result = model.dew_temperature(y_input, float(pressure_bar), T_bounds=(float(tmin), float(tmax)))
                st.success("Cálculo completado.")
                m1, m2 = st.columns(2)
                m1.metric("T de rocío", f"{result['T_K']:.3f} K")
                m2.metric("T de rocío", f"{result['T_K'] - 273.15:.3f} °C")
                st.markdown("**Composición de vapor normalizada (entrada)**")
                st.dataframe(composition_dataframe(result["y"], "y"), use_container_width=True)
                st.markdown("**Composición líquida en equilibrio (salida)**")
                st.dataframe(composition_dataframe(result["x"], "x"), use_container_width=True)
                st.markdown("**Constantes de equilibrio K**")
                k_df = pd.DataFrame({"Componente": list(COMPONENTS), "K": result["K"]})
                st.dataframe(k_df, use_container_width=True)
                st.download_button(
                    "Descargar resultado de rocío (JSON)",
                    json_download_bytes(serializable(result)),
                    file_name="punto_rocio_ternario.json",
                    mime="application/json",
                )
            except Exception as exc:
                st.error(f"No fue posible calcular el punto de rocío: {exc}")

    st.caption("Las composiciones se normalizan automáticamente si la suma ingresada es distinta de 1.")


def main() -> None:
    st.set_page_config(page_title="Equilibrio Líquido-Vapor", layout="wide")
    render_header()

    precomputed_data = load_precomputed_data()
    experimental_data = load_experimental_data()

    tab_binary, tab_ternary = st.tabs(
        ["Comparación binaria con base precalculada", "Mezcla ternaria: punto de burbuja y rocío"]
    )
    with tab_binary:
        render_binary_tab(precomputed_data, experimental_data)
    with tab_ternary:
        render_ternary_tab()


if __name__ == "__main__":
    main()
