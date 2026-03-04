"""
Demo Stability Patch
- Stabilized end-to-end workflow: upload -> validate -> lineage confirm -> stats -> ML -> AI summary.
- Removed hard dependency on local processed CSV and added dummy-data fallback.
- Added safe optional torch summary integration with graceful fallback.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    APP_TITLE: str = "Fatigue Data Intelligence Dashboard"
    APP_ICON: str = "🧠"
    RANDOM_SEED: int = 42


REQUIRED_COLUMNS = [
    "Specimen ID",
    "Process Type",
    "Yield Strength",
    "Melting Temperature",
    "Cycles to Failure",
    "Mean PSA",
    "Mean Stress",
    "Input Frequency",
    "Total Strain Amplitude",
    "Grain Size",
]

OPTIONAL_COLUMNS = [
    "UTS",
    "Elongation",
    "Hardness",
    "Mean Strain Amplitude",
    "Mean Unloading Modulus",
    "Fractographic Features",
]


CANONICAL_ALIASES = {
    "specimen id": "Specimen ID",
    "specimen_id": "Specimen ID",
    "id": "Specimen ID",
    "process type": "Process Type",
    "route_id": "Process Type",
    "route": "Process Type",
    "yield strength": "Yield Strength",
    "yield_strength": "Yield Strength",
    "ys": "Yield Strength",
    "melting temperature": "Melting Temperature",
    "melting_temp": "Melting Temperature",
    "temperature_c": "Melting Temperature",
    "cycles to failure": "Cycles to Failure",
    "cycles_to_failure": "Cycles to Failure",
    "nf": "Cycles to Failure",
    "mean psa": "Mean PSA",
    "psa": "Mean PSA",
    "mean stress": "Mean Stress",
    "mean_stress": "Mean Stress",
    "input frequency": "Input Frequency",
    "frequency": "Input Frequency",
    "frequency_hz": "Input Frequency",
    "total strain amplitude": "Total Strain Amplitude",
    "tsa": "Total Strain Amplitude",
    "tsa_percent": "Total Strain Amplitude",
    "grain size": "Grain Size",
    "grain_size": "Grain Size",
    "grain_size_um": "Grain Size",
    "uts": "UTS",
    "elongation": "Elongation",
    "hardness": "Hardness",
    "hardness_hv": "Hardness",
    "mean strain amplitude": "Mean Strain Amplitude",
    "mean unloading modulus": "Mean Unloading Modulus",
    "fractographic features": "Fractographic Features",
}


def init_session_state() -> None:
    defaults = {
        "current_data": None,
        "data_uploaded": False,
        "validation_report": {},
        "validation_confirmed": False,
        "stats_results": None,
        "ml_results": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def generate_dummy_fatigue_data(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    process_types = np.array(["T5", "T6A", "T6W", "DCT6", "ECAP90", "ECAP120"])

    df = pd.DataFrame(
        {
            "Specimen ID": [f"S{i+1:03d}" for i in range(n)],
            "Process Type": rng.choice(process_types, size=n),
            "Yield Strength": rng.normal(220, 30, n).clip(120, 350),
            "Melting Temperature": rng.normal(650, 10, n).clip(600, 700),
            "Mean PSA": rng.normal(0.35, 0.08, n).clip(0.1, 0.8),
            "Mean Stress": rng.normal(120, 25, n).clip(40, 220),
            "Input Frequency": rng.choice([0.1, 0.3, 0.5, 0.7, 1.0], size=n),
            "Total Strain Amplitude": rng.normal(0.4, 0.1, n).clip(0.1, 0.9),
            "Grain Size": rng.normal(35, 12, n).clip(5, 90),
            "Hardness": rng.normal(95, 15, n).clip(50, 160),
        }
    )
    base = (
        4500
        + 6 * df["Yield Strength"]
        - 9 * df["Mean Stress"]
        - 4200 * df["Total Strain Amplitude"]
        - 16 * df["Grain Size"]
        + rng.normal(0, 300, n)
    )
    df["Cycles to Failure"] = np.maximum(base, 150).round(0)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        normalized = col.strip().lower().replace("-", " ").replace("__", " ")
        normalized = " ".join(normalized.split())
        canonical = CANONICAL_ALIASES.get(normalized)
        if canonical:
            rename_map[col] = canonical
    normalized_df = df.rename(columns=rename_map).copy()
    return normalized_df


def build_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]

    numeric_expected = [c for c in REQUIRED_COLUMNS if c not in ["Specimen ID", "Process Type"]]
    dtype_issues = {}
    for col in numeric_expected:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues[col] = str(df[col].dtype)

    object_expected = ["Specimen ID", "Process Type"]
    for col in object_expected:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues[col] = str(df[col].dtype)

    missing_values = df.isna().sum().to_dict()

    return {
        "required_ok": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "dtype_issues": dtype_issues,
        "missing_values": missing_values,
        "row_count": len(df),
    }


def get_active_dataset() -> pd.DataFrame:
    data = st.session_state.get("current_data")
    if isinstance(data, pd.DataFrame) and not data.empty:
        return data
    df = generate_dummy_fatigue_data(seed=AppConfig.RANDOM_SEED)
    st.session_state["current_data"] = df
    st.session_state["data_uploaded"] = False
    if not st.session_state.get("validation_report"):
        st.session_state["validation_report"] = build_validation_report(df)
    return df


def try_get_torch_engine():
    try:
        from ml.torch_summary_engine import TorchSummaryEngine

        return TorchSummaryEngine()
    except Exception:
        return None


def render_route_kpis_and_torch_summary(df: pd.DataFrame) -> None:
    st.markdown("### Route KPIs")
    process_col = "Process Type" if "Process Type" in df.columns else None

    if process_col:
        routes = sorted(df[process_col].dropna().astype(str).unique().tolist())
        selected_route = st.selectbox("Select Process Type", options=routes, key="kpi_route")
        df_route = df[df[process_col].astype(str) == selected_route]
    else:
        selected_route = "All"
        df_route = df

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(df_route))
    mean_nf = df_route["Cycles to Failure"].mean() if "Cycles to Failure" in df_route.columns else np.nan
    c2.metric("Mean Cycles to Failure", f"{mean_nf:.1f}" if pd.notna(mean_nf) else "N/A")
    cov = (
        (df_route["Cycles to Failure"].std() / mean_nf) * 100
        if "Cycles to Failure" in df_route.columns and len(df_route) > 1 and pd.notna(mean_nf) and mean_nf != 0
        else np.nan
    )
    c3.metric("CoV (%)", f"{cov:.1f}" if pd.notna(cov) else "N/A")

    st.markdown("### PyTorch Model Summary (Optional)")
    engine = try_get_torch_engine()
    if engine is None:
        st.info("PyTorch summary engine unavailable. Using fallback AI summary path.")
        return

    try:
        summary = engine.generate_summary(selected_route=selected_route, df=df)
        st.write(summary)
    except Exception as exc:
        st.warning(f"PyTorch summary unavailable for this run: {type(exc).__name__}")


def generate_ai_summary(
    df: pd.DataFrame,
    stats: Optional[Dict[str, Any]] = None,
    ml: Optional[Dict[str, Any]] = None,
    user_query: Optional[str] = None,
) -> str:
    engine = try_get_torch_engine()
    if engine is not None:
        try:
            return engine.generate_summary(df=df, stats=stats, ml=ml, user_query=user_query)
        except Exception:
            pass

    rows = len(df)
    process_count = df["Process Type"].nunique() if "Process Type" in df.columns else 0
    missing_total = int(df.isna().sum().sum())

    lines = [
        "Fallback AI Summary (demo-safe):",
        f"- Dataset size: {rows} rows",
        f"- Distinct process routes: {process_count}",
        f"- Total missing values: {missing_total}",
    ]

    if stats:
        target_mean = stats.get("target_mean")
        target_cov = stats.get("target_cov")
        if target_mean is not None:
            lines.append(f"- Average Cycles to Failure: {target_mean:.2f}")
        if target_cov is not None:
            lines.append(f"- CoV of Cycles to Failure: {target_cov:.2f}%")
        ranking = stats.get("influence_ranking", [])
        if ranking:
            top = ranking[0]
            lines.append(f"- Strongest linear influence: {top['feature']} (|corr|={top['abs_corr']:.3f})")

    if ml:
        pred = ml.get("prediction")
        r2 = ml.get("r2")
        rmse = ml.get("rmse")
        if pred is not None:
            lines.append(f"- Predicted Cycles to Failure: {pred:.2f}")
        if r2 is not None and rmse is not None:
            lines.append(f"- Model fit: R²={r2:.3f}, RMSE={rmse:.2f}")
        else:
            lines.append("- Model fit: demo mode (insufficient data for reliable metrics).")

    if user_query:
        lines.append(f"- User query response: At a high level, '{user_query}' can be assessed from stress/strain and process effects shown above.")

    return "\n".join(lines)


def show_executive_dashboard() -> None:
    st.header("Executive Dashboard")

    st.markdown("### System Layer")
    c1, c2, c3 = st.columns(3)
    c1.success("Connection Status: Connected (demo)")
    c2.info("Subscription: Internal Demo License")
    c3.markdown("Help/Support: service@fdid.in")

    st.markdown("### CSV Upload + Preview")
    uploaded = st.file_uploader("Upload fatigue CSV", type=["csv"], accept_multiple_files=False)

    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            uploaded_df = normalize_columns(uploaded_df)
            report = build_validation_report(uploaded_df)
            st.session_state["current_data"] = uploaded_df
            st.session_state["data_uploaded"] = True
            st.session_state["validation_report"] = report
            st.session_state["validation_confirmed"] = False
            st.success(f"Loaded {len(uploaded_df)} rows from {uploaded.name}")
        except Exception as exc:
            st.error(f"Unable to parse CSV: {type(exc).__name__}")

    df = get_active_dataset()
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Schema Validation")
    if st.button("Run Schema Validation", use_container_width=True):
        st.session_state["validation_report"] = build_validation_report(df)

    report = st.session_state.get("validation_report") or build_validation_report(df)
    st.session_state["validation_report"] = report

    if report["missing_required"]:
        st.error(f"Missing required columns: {report['missing_required']}")
    else:
        st.success("All required columns are present.")

    if report["missing_optional"]:
        st.warning(f"Missing optional columns (allowed): {report['missing_optional']}")

    render_route_kpis_and_torch_summary(df)

    st.markdown("### AI Summary")
    user_query = st.text_input("Ask the AI assistant", placeholder="What affects fatigue life the most?")
    summary = generate_ai_summary(
        df,
        stats=st.session_state.get("stats_results"),
        ml=st.session_state.get("ml_results"),
        user_query=user_query if user_query else None,
    )
    st.text_area("Summary", value=summary, height=220)


def show_data_lineage() -> None:
    st.header("Data Lineage")
    df = get_active_dataset()
    report = st.session_state.get("validation_report") or build_validation_report(df)
    st.session_state["validation_report"] = report

    st.markdown("### Data Pipeline")
    st.markdown("Upload CSV → Normalize Columns → Validate Schema → Statistics → ML Prediction → AI Summary")

    st.markdown("### Data Type Validation")
    if report["dtype_issues"]:
        for col, dtype in report["dtype_issues"].items():
            st.warning(f"Datatype mismatch for `{col}` (found `{dtype}`).")
    else:
        st.success("No datatype mismatches detected for canonical expectations.")

    st.markdown("### Missing Values")
    missing_df = pd.DataFrame(
        [{"column": k, "missing": int(v)} for k, v in report["missing_values"].items() if int(v) > 0]
    )
    if missing_df.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing_df, use_container_width=True)

    if report["row_count"] < 10:
        st.warning("Insufficient rows for robust statistical trends (<10 rows).")

    if report["missing_optional"]:
        st.warning(f"Optional columns missing: {report['missing_optional']}")

    if report["missing_required"]:
        st.error("Cannot proceed until required columns are available.")

    if st.button("Confirm & Proceed", use_container_width=True):
        if report["required_ok"]:
            st.session_state["validation_confirmed"] = True
            st.success("Validation confirmed. Statistical Modelling and Machine Learning pages are unlocked.")
        else:
            st.session_state["validation_confirmed"] = False
            st.error("Validation failed due to missing required columns.")


def show_statistical_modelling() -> None:
    st.header("Statistical Modelling")
    if not st.session_state.get("validation_confirmed", False):
        st.warning("Please complete Data Lineage and click 'Confirm & Proceed' first.")
        st.stop()

    df = get_active_dataset().copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    st.markdown("### Descriptive Statistics")
    if not numeric_cols:
        st.warning("No numeric columns available.")
        st.stop()

    desc = df[numeric_cols].describe().T
    st.dataframe(desc, use_container_width=True)

    stats_results: Dict[str, Any] = {"descriptive": desc.to_dict()}

    st.markdown("### Distribution: Cycles to Failure")
    if "Cycles to Failure" in df.columns and pd.api.types.is_numeric_dtype(df["Cycles to Failure"]):
        hist = (
            alt.Chart(df)
            .mark_bar()
            .encode(x=alt.X("Cycles to Failure:Q", bin=True), y="count()")
            .properties(height=280)
        )
        st.altair_chart(hist, use_container_width=True)
    else:
        st.warning("'Cycles to Failure' unavailable for distribution plot.")

    st.markdown("### Correlation")
    corr = df[numeric_cols].corr(numeric_only=True)
    if corr.shape[0] >= 2:
        corr_long = corr.reset_index().melt(id_vars="index", var_name="feature", value_name="corr")
        heatmap = (
            alt.Chart(corr_long)
            .mark_rect()
            .encode(x="index:N", y="feature:N", color="corr:Q", tooltip=["index", "feature", "corr"])
            .properties(height=350)
        )
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.dataframe(corr, use_container_width=True)

    st.markdown("### Influence Ranking vs Cycles to Failure")
    ranking: List[Dict[str, Any]] = []
    if "Cycles to Failure" in corr.columns:
        target_corr = corr["Cycles to Failure"].drop(labels=["Cycles to Failure"], errors="ignore").dropna()
        if not target_corr.empty:
            rank_df = target_corr.abs().sort_values(ascending=False).reset_index()
            rank_df.columns = ["feature", "abs_corr"]
            st.dataframe(rank_df, use_container_width=True)
            ranking = rank_df.to_dict("records")

    target_series = df["Cycles to Failure"] if "Cycles to Failure" in df.columns else pd.Series(dtype=float)
    stats_results.update(
        {
            "target_mean": float(target_series.mean()) if not target_series.empty else None,
            "target_cov": float((target_series.std() / target_series.mean()) * 100)
            if not target_series.empty and target_series.mean() != 0
            else None,
            "influence_ranking": ranking,
        }
    )

    st.session_state["stats_results"] = stats_results


def show_machine_learning() -> None:
    st.header("Machine Learning")
    if not st.session_state.get("validation_confirmed", False):
        st.warning("Please complete Data Lineage and click 'Confirm & Proceed' first.")
        st.stop()

    df = get_active_dataset().copy()
    required_ml_cols = [
        "Yield Strength",
        "Mean Stress",
        "Mean PSA",
        "Input Frequency",
        "Total Strain Amplitude",
        "Grain Size",
        "Cycles to Failure",
    ]
    missing_ml = [c for c in required_ml_cols if c not in df.columns]
    if missing_ml:
        st.warning(f"Insufficient columns for ML baseline: {missing_ml}")
        st.stop()

    work = df.copy()
    work["Process Type Enc"] = pd.factorize(work.get("Process Type", "Unknown"))[0]

    feature_cols = [
        "Grain Size",
        "Process Type Enc",
        "Yield Strength",
        "Mean Stress",
        "Mean PSA",
        "Input Frequency",
        "Total Strain Amplitude",
    ]

    X = work[feature_cols].to_numpy(dtype=float)
    y = work["Cycles to Failure"].to_numpy(dtype=float)

    def fit_linear_np(X_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
        X_aug = np.c_[np.ones(len(X_fit)), X_fit]
        coef, *_ = np.linalg.lstsq(X_aug, y_fit, rcond=None)
        return coef

    def predict_linear_np(coef: np.ndarray, X_pred: np.ndarray) -> np.ndarray:
        X_aug = np.c_[np.ones(len(X_pred)), X_pred]
        return X_aug @ coef

    metrics = {"r2": None, "rmse": None}

    if len(work) >= 12:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(work))
        split = max(1, int(0.8 * len(work)))
        train_idx, test_idx = idx[:split], idx[split:]
        coef = fit_linear_np(X[train_idx], y[train_idx])
        pred_test = predict_linear_np(coef, X[test_idx]) if len(test_idx) > 0 else np.array([])
        if len(test_idx) > 0:
            y_test = y[test_idx]
            ss_res = float(np.sum((y_test - pred_test) ** 2))
            ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
            metrics["r2"] = float(1 - ss_res / ss_tot) if ss_tot != 0 else None
            metrics["rmse"] = float(np.sqrt(np.mean((y_test - pred_test) ** 2)))
    else:
        coef = fit_linear_np(X, y)

    st.markdown("### Input Controls")
    process_options = sorted(df["Process Type"].astype(str).unique().tolist()) if "Process Type" in df.columns else ["Unknown"]
    col1, col2 = st.columns(2)
    with col1:
        grain_size = st.number_input("Grain Size", value=float(df["Grain Size"].median()))
        process_type = st.selectbox("Process Type", options=process_options)
        yield_strength = st.number_input("Yield Strength", value=float(df["Yield Strength"].median()))
        mean_stress = st.number_input("Mean Stress", value=float(df["Mean Stress"].median()))
    with col2:
        mean_psa = st.number_input("Mean PSA", value=float(df["Mean PSA"].median()))
        input_frequency = st.number_input("Input Frequency", value=float(df["Input Frequency"].median()))
        tsa_default = float(df["Total Strain Amplitude"].median()) if "Total Strain Amplitude" in df.columns else 0.4
        total_strain_amplitude = st.number_input("Total Strain Amplitude", value=tsa_default)

    proc_code = process_options.index(process_type) if process_type in process_options else 0
    sample = pd.DataFrame(
        [[grain_size, proc_code, yield_strength, mean_stress, mean_psa, input_frequency, total_strain_amplitude]],
        columns=feature_cols,
    )
    prediction = float(predict_linear_np(coef, sample.to_numpy(dtype=float))[0])

    st.success(f"Predicted Cycles to Failure: {prediction:.2f}")

    if metrics["r2"] is not None and metrics["rmse"] is not None:
        st.write(f"Train/Test Metrics: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.2f}")
    else:
        st.info("Metrics: demo mode (not enough rows for robust split).")

    st.markdown("### PyTorch model (optional)")
    if try_get_torch_engine() is None:
        st.info("PyTorch model unavailable in this environment; baseline sklearn model is active.")
    else:
        st.caption("PyTorch engine detected. Optional integration can be enabled without impacting baseline flow.")

    st.session_state["ml_results"] = {
        "prediction": prediction,
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "model": "LinearRegression (NumPy baseline)",
    }


def main() -> None:
    st.set_page_config(page_title=AppConfig.APP_TITLE, page_icon=AppConfig.APP_ICON, layout="wide")
    init_session_state()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Executive Dashboard", "Data Lineage", "Statistical Modelling", "Machine Learning"],
        index=0,
    )

    if page == "Executive Dashboard":
        show_executive_dashboard()
    elif page == "Data Lineage":
        show_data_lineage()
    elif page == "Statistical Modelling":
        show_statistical_modelling()
    elif page == "Machine Learning":
        show_machine_learning()


if __name__ == "__main__":
    main()
