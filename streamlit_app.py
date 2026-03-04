"""
Demo Stability Patch
- Updated schema handling to new canonical fatigue CSV format with backward-compatible alias mapping.
- Restored top bar controls (license/connection/new project/help) and bottom-right live status panel.
- Preserved demo-safe end-to-end workflow with validation gating, stats, ML prediction, and AI summary fallback.
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


CANONICAL_COLUMNS = [
    "specimen_id",
    "route_family",
    "process_subtype",
    "soak_hours",
    "ecap_angle_deg",
    "YS_MPa",
    "UTS_MPa",
    "elongation_percent",
    "Hardness_hv",
    "grain_size_um",
    "Nf",
    "TSA",
    "frequency_Hz",
    "temperature_C",
    "log_Nf",
    "d_inv_sqrt",
    "strength_ratio",
    "fatigue_efficiency",
    "PSA_mean",
    "mean_stress_mean",
    "stress_amp_mean",
    "unloading_modulus_mean",
]

REQUIRED_COLUMNS = [
    "specimen_id",
    "route_family",
    "YS_MPa",
    "grain_size_um",
    "Nf",
    "TSA",
    "frequency_Hz",
    "PSA_mean",
    "mean_stress_mean",
]

OPTIONAL_COLUMNS = [c for c in CANONICAL_COLUMNS if c not in REQUIRED_COLUMNS]


ALIAS_MAP = {
    # new/canonical variants
    "specimen id": "specimen_id",
    "specimen_id": "specimen_id",
    "sample_id": "specimen_id",
    "route family": "route_family",
    "route_family": "route_family",
    "route": "route_family",
    "process type": "route_family",
    "process_type": "route_family",
    "process subtype": "process_subtype",
    "process_subtype": "process_subtype",
    "soak hours": "soak_hours",
    "soak_hours": "soak_hours",
    "ecap angle deg": "ecap_angle_deg",
    "ecap_angle_deg": "ecap_angle_deg",
    "ys mpa": "YS_MPa",
    "ys_mpa": "YS_MPa",
    "yield strength": "YS_MPa",
    "yield_strength": "YS_MPa",
    "uts mpa": "UTS_MPa",
    "uts_mpa": "UTS_MPa",
    "uts": "UTS_MPa",
    "elongation percent": "elongation_percent",
    "elongation_percent": "elongation_percent",
    "elongation": "elongation_percent",
    "hardness hv": "Hardness_hv",
    "hardness_hv": "Hardness_hv",
    "hardness": "Hardness_hv",
    "grain size um": "grain_size_um",
    "grain_size_um": "grain_size_um",
    "grain size": "grain_size_um",
    "nf": "Nf",
    "cycles to failure": "Nf",
    "cycles_to_failure": "Nf",
    "tsa": "TSA",
    "total strain amplitude": "TSA",
    "total_strain_amplitude": "TSA",
    "frequency hz": "frequency_Hz",
    "frequency_hz": "frequency_Hz",
    "input frequency": "frequency_Hz",
    "temperature c": "temperature_C",
    "temperature_c": "temperature_C",
    "log nf": "log_Nf",
    "log_nf": "log_Nf",
    "d inv sqrt": "d_inv_sqrt",
    "d_inv_sqrt": "d_inv_sqrt",
    "strength ratio": "strength_ratio",
    "strength_ratio": "strength_ratio",
    "fatigue efficiency": "fatigue_efficiency",
    "fatigue_efficiency": "fatigue_efficiency",
    "psa mean": "PSA_mean",
    "psa_mean": "PSA_mean",
    "mean psa": "PSA_mean",
    "mean stress mean": "mean_stress_mean",
    "mean_stress_mean": "mean_stress_mean",
    "mean stress": "mean_stress_mean",
    "stress amp mean": "stress_amp_mean",
    "stress_amp_mean": "stress_amp_mean",
    "unloading modulus mean": "unloading_modulus_mean",
    "unloading_modulus_mean": "unloading_modulus_mean",
    # old dashboard names mapped into new canonical
    "specimen": "specimen_id",
    "specimenid": "specimen_id",
    "process": "route_family",
    "route_id": "route_family",
    "yield strength mpa": "YS_MPa",
    "melting temperature": "temperature_C",
    "cycles": "Nf",
    "mean psa old": "PSA_mean",
    "mean stress old": "mean_stress_mean",
    "input frequency hz": "frequency_Hz",
    "grain size": "grain_size_um",
}


def init_session_state() -> None:
    defaults = {
        "current_data": None,
        "data_uploaded": False,
        "validation_report": {},
        "validation_confirmed": False,
        "stats_results": None,
        "ml_results": None,
        "last_operation": "App started",
        "db_connected": False,
        "license_tier": "Free Demo",
        "connection_mode": "Local",
        "current_project_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def connect_to_database() -> bool:
    try:
        mode = st.session_state.get("connection_mode", "Local")
        if mode == "Local":
            st.session_state["db_connected"] = True
            return True
        # demo-safe simulated server attempt
        _ = "demo server handshake"
        st.session_state["db_connected"] = True
        return True
    except Exception:
        st.session_state["db_connected"] = False
        return False


def generate_dummy_fatigue_data(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    routes = np.array(["T5", "T6A", "T6W", "DCT6", "ECAP90", "ECAP120"])
    subtypes = np.array(["Base", "Aged", "ECAP", "Hybrid"])

    df = pd.DataFrame(
        {
            "specimen_id": [f"S{i+1:03d}" for i in range(n)],
            "route_family": rng.choice(routes, n),
            "process_subtype": rng.choice(subtypes, n),
            "soak_hours": rng.uniform(1, 18, n).round(2),
            "ecap_angle_deg": rng.choice([90, 120], n),
            "YS_MPa": rng.normal(220, 28, n).clip(120, 350),
            "UTS_MPa": rng.normal(280, 30, n).clip(150, 420),
            "elongation_percent": rng.uniform(4, 20, n).round(2),
            "Hardness_hv": rng.normal(95, 15, n).clip(50, 170),
            "grain_size_um": rng.normal(35, 12, n).clip(5, 95),
            "TSA": rng.normal(0.40, 0.10, n).clip(0.1, 0.9),
            "frequency_Hz": rng.choice([0.1, 0.3, 0.5, 0.7, 1.0], n),
            "temperature_C": rng.normal(25, 5, n).clip(10, 80),
            "PSA_mean": rng.normal(0.32, 0.09, n).clip(0.05, 0.9),
            "mean_stress_mean": rng.normal(120, 25, n).clip(30, 220),
            "stress_amp_mean": rng.normal(80, 15, n).clip(20, 180),
            "unloading_modulus_mean": rng.normal(60, 10, n).clip(20, 120),
        }
    )

    nf = (
        4200
        + 6.5 * df["YS_MPa"]
        - 8.0 * df["mean_stress_mean"]
        - 3900 * df["TSA"]
        - 14 * df["grain_size_um"]
        + rng.normal(0, 280, n)
    )
    df["Nf"] = np.maximum(150, nf).round(0)
    df["log_Nf"] = np.log10(df["Nf"])
    df["d_inv_sqrt"] = (1 / np.sqrt(df["grain_size_um"])).round(5)
    df["strength_ratio"] = (df["YS_MPa"] / df["UTS_MPa"]).round(4)
    df["fatigue_efficiency"] = (df["Nf"] / df["Nf"].max()).round(5)
    return df[CANONICAL_COLUMNS]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        key = col.strip().lower().replace("-", " ").replace("/", " ").replace("__", "_")
        key = " ".join(key.replace("_", " ").split())
        canonical = ALIAS_MAP.get(key)
        if canonical is not None:
            rename_map[col] = canonical

    norm = df.rename(columns=rename_map).copy()
    return norm


def build_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]

    dtype_issues: Dict[str, str] = {}
    numeric_required = [c for c in REQUIRED_COLUMNS if c not in {"specimen_id", "route_family"}]
    for col in numeric_required:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues[col] = str(df[col].dtype)

    for col in ["specimen_id", "route_family", "process_subtype"]:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues[col] = str(df[col].dtype)

    return {
        "required_ok": len(missing_required) == 0,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "dtype_issues": dtype_issues,
        "missing_values": df.isna().sum().to_dict(),
        "row_count": int(len(df)),
    }


def get_active_dataset() -> pd.DataFrame:
    data = st.session_state.get("current_data")
    if isinstance(data, pd.DataFrame) and not data.empty:
        return data

    df = generate_dummy_fatigue_data(seed=AppConfig.RANDOM_SEED)
    st.session_state["current_data"] = df
    st.session_state["data_uploaded"] = False
    st.session_state["validation_report"] = build_validation_report(df)
    return df


def try_get_torch_engine():
    try:
        from ml.torch_summary_engine import TorchSummaryEngine

        return TorchSummaryEngine()
    except Exception:
        return None


def render_top_bar() -> None:
    st.markdown("## Fatigue Data Intelligence Dashboard")
    col1, col2, col3, col4, col5 = st.columns([1.1, 1.1, 1.1, 1.4, 1.2])

    with col1:
        st.session_state["license_tier"] = st.selectbox(
            "License",
            ["Free Demo", "Academic", "Industry", "Enterprise"],
            index=["Free Demo", "Academic", "Industry", "Enterprise"].index(st.session_state.get("license_tier", "Free Demo")),
            key="license_tier_selector",
        )

    with col2:
        st.session_state["connection_mode"] = st.selectbox(
            "Connection",
            ["Local", "Server"],
            index=["Local", "Server"].index(st.session_state.get("connection_mode", "Local")),
            key="connection_mode_selector",
        )

    with col3:
        if st.button("➕ New Project", use_container_width=True):
            st.session_state["current_project_id"] = f"project_{np.random.randint(1000, 9999)}"
            st.session_state["current_data"] = None
            st.session_state["data_uploaded"] = False
            st.session_state["validation_confirmed"] = False
            st.session_state["stats_results"] = None
            st.session_state["ml_results"] = None
            st.session_state["last_operation"] = "New project initialized"
            st.success("New project started.")

    with col4:
        st.markdown(
            f"**Project:** {st.session_state.get('current_project_id') or 'None'}  \\\n**Mode:** {st.session_state.get('connection_mode', 'Local')}"
        )

    with col5:
        with st.expander("❓ Help / Support"):
            st.markdown("📧 service@fdid.in")
            st.markdown("💬 Demo support available during business hours")


def render_status_panel() -> None:
    status_color = "#4CAF50" if st.session_state.get("db_connected", False) else "#FF5252"
    status_text = "Connected" if st.session_state.get("db_connected", False) else "Not connected"

    html = f"""
    <div style="
        position: fixed;
        bottom: 12px;
        right: 12px;
        background-color: #0E1117;
        color: white;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px 12px;
        z-index: 9999;
        width: 270px;
        font-size: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    ">
        <div style="font-weight:700; color:#64B5F6; margin-bottom:6px;">Status</div>
        <div><b>DB:</b> <span style="color:{status_color};">● {status_text}</span></div>
        <div><b>Connection:</b> {st.session_state.get('connection_mode', 'Local')}</div>
        <div><b>License:</b> {st.session_state.get('license_tier', 'Free Demo')}</div>
        <div style="margin-top:4px;"><b>Last op:</b><br><span style="color:#c7c7c7;">{st.session_state.get('last_operation', 'App started')}</span></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_route_kpis_and_torch_summary(df: pd.DataFrame) -> None:
    st.markdown("### Route KPIs")
    route_options = sorted(df["route_family"].dropna().astype(str).unique()) if "route_family" in df.columns else ["All"]
    selected_route = st.selectbox("Select route family", route_options, key="route_selector")
    df_route = df[df["route_family"].astype(str) == selected_route] if "route_family" in df.columns else df

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(df_route))
    mean_nf = df_route["Nf"].mean() if "Nf" in df_route.columns else np.nan
    c2.metric("Mean Nf", f"{mean_nf:.1f}" if pd.notna(mean_nf) else "N/A")
    cov_nf = (df_route["Nf"].std() / mean_nf * 100) if "Nf" in df_route.columns and len(df_route) > 1 and mean_nf else np.nan
    c3.metric("CoV (%)", f"{cov_nf:.1f}" if pd.notna(cov_nf) else "N/A")

    st.markdown("### PyTorch Summary (Optional)")
    engine = try_get_torch_engine()
    if engine is None:
        st.info("PyTorch summary engine unavailable; fallback summary remains active.")
        return
    try:
        st.write(engine.generate_summary(selected_route=selected_route, df=df))
    except Exception as exc:
        st.warning(f"PyTorch summary failed safely: {type(exc).__name__}")


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
    routes = int(df["route_family"].nunique()) if "route_family" in df.columns else 0
    missing_count = int(df.isna().sum().sum())

    lines = [
        "Fallback AI Summary (demo-safe)",
        f"- Dataset rows: {rows}",
        f"- Distinct routes: {routes}",
        f"- Total missing values: {missing_count}",
    ]

    if stats:
        if stats.get("target_mean") is not None:
            lines.append(f"- Mean Nf: {stats['target_mean']:.2f}")
        if stats.get("target_cov") is not None:
            lines.append(f"- CoV(Nf): {stats['target_cov']:.2f}%")
        ranking = stats.get("influence_ranking") or []
        if ranking:
            lines.append(f"- Top influence vs Nf: {ranking[0]['feature']} (|corr|={ranking[0]['abs_corr']:.3f})")

    if ml:
        if ml.get("prediction") is not None:
            lines.append(f"- ML predicted Nf: {ml['prediction']:.2f}")
        if ml.get("r2") is not None and ml.get("rmse") is not None:
            lines.append(f"- Baseline fit: R²={ml['r2']:.3f}, RMSE={ml['rmse']:.2f}")
        else:
            lines.append("- Baseline fit: demo mode")

    if user_query:
        lines.append(f"- Query response: '{user_query}' can be interpreted using route-family and stress/strain indicators above.")

    return "\n".join(lines)


def show_executive_dashboard() -> None:
    render_top_bar()
    connect_to_database()

    st.header("Executive Dashboard")
    st.markdown("### System Layer")
    c1, c2, c3 = st.columns(3)
    c1.success(f"Connection status: {'Connected' if st.session_state.get('db_connected') else 'Not connected'}")
    c2.info(f"License: {st.session_state.get('license_tier', 'Free Demo')}")
    c3.info(f"Connection Mode: {st.session_state.get('connection_mode', 'Local')}")

    st.markdown("### CSV Upload + Preview")
    upload = st.file_uploader("Upload fatigue CSV", type=["csv"], accept_multiple_files=False)
    if upload is not None:
        try:
            data = pd.read_csv(upload)
            data = normalize_columns(data)
            st.session_state["current_data"] = data
            st.session_state["data_uploaded"] = True
            st.session_state["validation_report"] = build_validation_report(data)
            st.session_state["validation_confirmed"] = False
            st.session_state["last_operation"] = f"CSV uploaded: {upload.name}"
            st.success(f"Loaded {len(data)} rows.")
        except Exception as exc:
            st.error(f"Unable to parse CSV: {type(exc).__name__}")

    df = get_active_dataset()
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Schema Validation")
    if st.button("Run Schema Validation", use_container_width=True):
        st.session_state["validation_report"] = build_validation_report(df)
        st.session_state["last_operation"] = "Schema validation executed"

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
    user_query = st.text_input("Ask the AI assistant", placeholder="Which variables most influence Nf?")
    if st.button("Generate AI Summary", use_container_width=True):
        summary = generate_ai_summary(
            df,
            stats=st.session_state.get("stats_results"),
            ml=st.session_state.get("ml_results"),
            user_query=user_query if user_query else None,
        )
        st.session_state["ai_summary_text"] = summary
        st.session_state["last_operation"] = "AI summary generated"

    summary_value = st.session_state.get("ai_summary_text", "Click 'Generate AI Summary' to produce a summary.")
    st.text_area("Summary", value=summary_value, height=220)


def show_data_lineage() -> None:
    st.header("Data Lineage")
    connect_to_database()

    df = get_active_dataset()
    report = st.session_state.get("validation_report") or build_validation_report(df)
    st.session_state["validation_report"] = report

    st.markdown("Upload CSV → Data Validation → Data Lineage → Statistical Modelling → ML Prediction → AI Summary")

    st.markdown("### Schema Checklist")
    req_df = pd.DataFrame({"required_columns": REQUIRED_COLUMNS, "present": [c in df.columns for c in REQUIRED_COLUMNS]})
    opt_df = pd.DataFrame({"optional_columns": OPTIONAL_COLUMNS, "present": [c in df.columns for c in OPTIONAL_COLUMNS]})
    st.dataframe(req_df, use_container_width=True)
    st.dataframe(opt_df, use_container_width=True)

    st.markdown("### Datatype Validation")
    if report["dtype_issues"]:
        for col, dtype in report["dtype_issues"].items():
            st.warning(f"Datatype mismatch for `{col}` (found `{dtype}`).")
    else:
        st.success("No datatype mismatches detected for expected canonical types.")

    st.markdown("### Missing Values")
    missing_df = pd.DataFrame([
        {"column": col, "missing": int(cnt)}
        for col, cnt in report["missing_values"].items()
        if int(cnt) > 0
    ])
    if missing_df.empty:
        st.success("No missing values detected.")
    else:
        st.dataframe(missing_df, use_container_width=True)

    if report["row_count"] < 10:
        st.warning("Insufficient rows for robust statistical analysis (<10 rows).")

    if report["missing_optional"]:
        st.warning(f"Optional columns missing: {report['missing_optional']}")

    if report["missing_required"]:
        st.error("Required columns missing. Cannot confirm validation.")

    if st.button("Confirm & Proceed", use_container_width=True):
        if report["required_ok"]:
            st.session_state["validation_confirmed"] = True
            st.session_state["last_operation"] = "Validation confirmed"
            st.success("Validation confirmed. You can proceed to Statistical Modelling and Machine Learning.")
        else:
            st.session_state["validation_confirmed"] = False
            st.error("Validation failed due to missing required columns.")


def show_statistical_modelling() -> None:
    st.header("Statistical Modelling")
    connect_to_database()

    if not st.session_state.get("validation_confirmed", False):
        st.warning("Please complete Data Lineage and click 'Confirm & Proceed' first.")
        st.stop()

    df = get_active_dataset().copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        st.warning("No numeric columns available.")
        st.stop()

    st.markdown("### Descriptive Statistics")
    desc = df[numeric_cols].describe().T
    st.dataframe(desc, use_container_width=True)

    st.markdown("### Distribution of Nf")
    if "Nf" in df.columns and pd.api.types.is_numeric_dtype(df["Nf"]):
        hist = alt.Chart(df).mark_bar().encode(x=alt.X("Nf:Q", bin=True), y="count()")
        st.altair_chart(hist, use_container_width=True)
    else:
        st.warning("Nf unavailable for distribution plot.")

    st.markdown("### Correlation")
    corr = df[numeric_cols].corr(numeric_only=True)
    if corr.shape[0] >= 2:
        corr_long = corr.reset_index().melt(id_vars="index", var_name="feature", value_name="corr")
        heatmap = alt.Chart(corr_long).mark_rect().encode(
            x="index:N", y="feature:N", color="corr:Q", tooltip=["index", "feature", "corr"]
        )
        st.altair_chart(heatmap, use_container_width=True)
    else:
        st.dataframe(corr, use_container_width=True)

    st.markdown("### Influence Ranking vs Nf")
    ranking: List[Dict[str, Any]] = []
    if "Nf" in corr.columns:
        target_corr = corr["Nf"].drop(labels=["Nf"], errors="ignore").dropna().abs().sort_values(ascending=False)
        if not target_corr.empty:
            rank_df = target_corr.reset_index()
            rank_df.columns = ["feature", "abs_corr"]
            ranking = rank_df.to_dict("records")
            st.dataframe(rank_df, use_container_width=True)

    nf = df["Nf"] if "Nf" in df.columns else pd.Series(dtype=float)
    st.session_state["stats_results"] = {
        "descriptive": desc.to_dict(),
        "target_mean": float(nf.mean()) if not nf.empty else None,
        "target_cov": float((nf.std() / nf.mean()) * 100) if (not nf.empty and nf.mean() != 0) else None,
        "influence_ranking": ranking,
    }
    st.session_state["last_operation"] = "Statistical modelling computed"


def show_machine_learning() -> None:
    st.header("Machine Learning")
    connect_to_database()

    if not st.session_state.get("validation_confirmed", False):
        st.warning("Please complete Data Lineage and click 'Confirm & Proceed' first.")
        st.stop()

    df = get_active_dataset().copy()
    required_for_ml = ["YS_MPa", "grain_size_um", "Nf", "TSA", "frequency_Hz", "PSA_mean", "mean_stress_mean"]
    missing = [c for c in required_for_ml if c not in df.columns]
    if missing:
        st.warning(f"Missing required ML columns: {missing}")
        st.stop()

    work = df.copy()
    work["route_family_code"] = pd.factorize(work.get("route_family", "Unknown"))[0]

    feature_cols = [
        "grain_size_um",
        "route_family_code",
        "YS_MPa",
        "mean_stress_mean",
        "PSA_mean",
        "frequency_Hz",
        "TSA",
    ]

    X = work[feature_cols].to_numpy(dtype=float)
    y = work["Nf"].to_numpy(dtype=float)

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
        if len(test_idx) > 0:
            preds = predict_linear_np(coef, X[test_idx])
            y_test = y[test_idx]
            ss_res = float(np.sum((y_test - preds) ** 2))
            ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
            metrics["r2"] = float(1 - ss_res / ss_tot) if ss_tot != 0 else None
            metrics["rmse"] = float(np.sqrt(np.mean((y_test - preds) ** 2)))
    else:
        coef = fit_linear_np(X, y)

    st.markdown("### Input Controls")
    route_options = sorted(df["route_family"].astype(str).unique().tolist()) if "route_family" in df.columns else ["Unknown"]
    left, right = st.columns(2)
    with left:
        grain_size = st.number_input("grain_size_um", value=float(df["grain_size_um"].median()))
        route_family = st.selectbox("route_family", options=route_options)
        ys_mpa = st.number_input("YS_MPa", value=float(df["YS_MPa"].median()))
        mean_stress = st.number_input("mean_stress_mean", value=float(df["mean_stress_mean"].median()))
    with right:
        psa_mean = st.number_input("PSA_mean", value=float(df["PSA_mean"].median()))
        freq = st.number_input("frequency_Hz", value=float(df["frequency_Hz"].median()))
        tsa = st.number_input("TSA", value=float(df["TSA"].median()))

    route_code = route_options.index(route_family) if route_family in route_options else 0
    sample = np.array([[grain_size, route_code, ys_mpa, mean_stress, psa_mean, freq, tsa]], dtype=float)
    prediction = float(predict_linear_np(coef, sample)[0])

    st.success(f"Predicted Nf: {prediction:.2f}")
    if metrics["r2"] is not None and metrics["rmse"] is not None:
        st.write(f"Metrics: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}")
    else:
        st.info("Metrics: demo mode")

    st.markdown("### PyTorch model (optional)")
    if try_get_torch_engine() is None:
        st.info("PyTorch unavailable. Baseline NumPy regression used.")
    else:
        st.caption("PyTorch engine detected. Optional model integration can be enabled safely.")

    st.session_state["ml_results"] = {
        "prediction": prediction,
        "r2": metrics["r2"],
        "rmse": metrics["rmse"],
        "model": "LinearRegression (NumPy baseline)",
    }
    st.session_state["last_operation"] = "ML prediction computed"


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

    render_status_panel()


if __name__ == "__main__":
    main()
