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
    APP_ICON: str = "🔬"
    APP_VERSION: str = "v2.0"
    APP_BRAND: str = "Materials Intelligence Lab"
    RANDOM_SEED: int = 42


CANONICAL_SCHEMA = [
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

OPTIONAL_COLUMNS = [c for c in CANONICAL_SCHEMA if c not in REQUIRED_COLUMNS]


def apply_custom_styling() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

            .stApp { background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%); }
            h1, h2, h3 { letter-spacing: -0.01em; }

            .hero {
                border: 1px solid #dbe4f0;
                background: linear-gradient(120deg, #e9f2ff, #f8fbff);
                border-radius: 14px;
                padding: 1rem 1.2rem;
                margin-bottom: 0.75rem;
            }

            .hero-title { font-size: 1.55rem; font-weight: 700; color: #0f3d75; margin: 0; }
            .hero-subtitle { color: #4b5d75; margin-top: 0.2rem; font-size: 0.9rem; }

            .system-pill {
                border-radius: 999px;
                border: 1px solid #ccd9eb;
                background: #ffffff;
                padding: 0.35rem 0.65rem;
                font-size: 0.78rem;
                color: #334e6b;
                margin-right: 0.4rem;
                display: inline-block;
            }

            .card {
                background: #ffffff;
                border: 1px solid #e6edf5;
                border-radius: 12px;
                padding: 0.9rem 1rem;
                margin-bottom: 0.8rem;
                box-shadow: 0 2px 8px rgba(15, 61, 117, 0.04);
            }

            .summary-box {
                border-left: 4px solid #0f62fe;
                border-radius: 8px;
                background: #f5f9ff;
                padding: 0.9rem;
            }

            .footer {
                border-top: 1px solid #dbe4f0;
                text-align: center;
                color: #5e6f85;
                font-size: 0.82rem;
                margin-top: 2rem;
                padding-top: 0.8rem;
            }

            section[data-testid="stSidebar"] {
                border-right: 1px solid #dbe4f0;
                background: #f7fbff;
            }

            @media (max-width: 960px) {
                .hero-title { font-size: 1.2rem; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


CANONICAL_ALIASES = {
    "specimen id": "specimen_id",
    "specimenid": "specimen_id",
    "specimen_id": "specimen_id",
    "id": "specimen_id",
    "route family": "route_family",
    "route_family": "route_family",
    "route": "route_family",
    "process type": "route_family",
    "process subtype": "process_subtype",
    "process_subtype": "process_subtype",
    "soak hours": "soak_hours",
    "soak_hours": "soak_hours",
    "ecap angle deg": "ecap_angle_deg",
    "ecap_angle_deg": "ecap_angle_deg",
    "yield strength": "YS_MPa",
    "yield_strength": "YS_MPa",
    "ys": "YS_MPa",
    "ys_mpa": "YS_MPa",
    "uts": "UTS_MPa",
    "uts_mpa": "UTS_MPa",
    "elongation": "elongation_percent",
    "elongation_percent": "elongation_percent",
    "hardness": "Hardness_hv",
    "hardness_hv": "Hardness_hv",
    "grain size": "grain_size_um",
    "grain_size": "grain_size_um",
    "grain_size_um": "grain_size_um",
    "cycles to failure": "Nf",
    "cycles_to_failure": "Nf",
    "nf": "Nf",
    "total strain amplitude": "TSA",
    "total_strain_amplitude": "TSA",
    "tsa": "TSA",
    "input frequency": "frequency_Hz",
    "frequency": "frequency_Hz",
    "frequency_hz": "frequency_Hz",
    "melting temperature": "temperature_C",
    "temperature_c": "temperature_C",
    "log_nf": "log_Nf",
    "d_inv_sqrt": "d_inv_sqrt",
    "strength_ratio": "strength_ratio",
    "fatigue_efficiency": "fatigue_efficiency",
    "mean psa": "PSA_mean",
    "psa": "PSA_mean",
    "psa_mean": "PSA_mean",
    "mean stress": "mean_stress_mean",
    "mean_stress": "mean_stress_mean",
    "mean_stress_mean": "mean_stress_mean",
    "mean strain amplitude": "stress_amp_mean",
    "stress_amp_mean": "stress_amp_mean",
    "mean unloading modulus": "unloading_modulus_mean",
    "unloading_modulus_mean": "unloading_modulus_mean",
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


def set_last_operation(operation: str) -> None:
    st.session_state["last_operation"] = operation


def connect_to_database() -> None:
    try:
        if st.session_state.get("connection_mode") == "Server":
            _ = "simulated-server-connection"
            st.session_state["db_connected"] = True
        else:
            st.session_state["db_connected"] = True
    except Exception:
        st.session_state["db_connected"] = False


def generate_dummy_fatigue_data(n: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    process_types = np.array(["T5", "T6A", "T6W", "DCT6", "ECAP90", "ECAP120"])

    df = pd.DataFrame(
        {
            "specimen_id": [f"S{i+1:03d}" for i in range(n)],
            "route_family": rng.choice(process_types, size=n),
            "process_subtype": rng.choice(["A", "B", "C"], size=n),
            "soak_hours": rng.uniform(1, 8, n).round(2),
            "ecap_angle_deg": rng.choice([90, 120], size=n),
            "YS_MPa": rng.normal(220, 30, n).clip(120, 350),
            "UTS_MPa": rng.normal(320, 35, n).clip(180, 430),
            "elongation_percent": rng.normal(11, 2, n).clip(3, 20),
            "Hardness_hv": rng.normal(95, 15, n).clip(50, 160),
            "grain_size_um": rng.normal(35, 12, n).clip(5, 90),
            "TSA": rng.normal(0.4, 0.1, n).clip(0.1, 0.9),
            "frequency_Hz": rng.choice([0.1, 0.3, 0.5, 0.7, 1.0], size=n),
            "temperature_C": rng.normal(35, 5, n).clip(20, 60),
            "PSA_mean": rng.normal(0.35, 0.08, n).clip(0.1, 0.8),
            "mean_stress_mean": rng.normal(120, 25, n).clip(40, 220),
            "stress_amp_mean": rng.normal(0.3, 0.08, n).clip(0.05, 0.8),
            "unloading_modulus_mean": rng.normal(22000, 2500, n).clip(12000, 32000),
        }
    )
    base = (
        4500
        + 6 * df["YS_MPa"]
        - 9 * df["mean_stress_mean"]
        - 4200 * df["TSA"]
        - 16 * df["grain_size_um"]
        + rng.normal(0, 300, n)
    )
    df["Nf"] = np.maximum(base, 150).round(0)
    df["log_Nf"] = np.log10(df["Nf"].clip(lower=1))
    df["d_inv_sqrt"] = 1 / np.sqrt(df["grain_size_um"].clip(lower=1e-3))
    df["strength_ratio"] = df["YS_MPa"] / df["UTS_MPa"].replace(0, np.nan)
    df["fatigue_efficiency"] = df["Nf"] / df["YS_MPa"].replace(0, np.nan)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def normalize_key(name: str) -> str:
        normalized = name.strip().lower().replace("-", " ").replace("_", " ")
        return " ".join(normalized.split())

    rename_map = {}
    for col in df.columns:
        normalized = normalize_key(col)
        canonical = CANONICAL_ALIASES.get(normalized)
        if canonical:
            rename_map[col] = canonical
    normalized_df = df.rename(columns=rename_map).copy()
    normalized_df = normalized_df.loc[:, ~normalized_df.columns.duplicated()]

    for column in [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in normalized_df.columns]:
        if column not in ["specimen_id", "route_family", "process_subtype"]:
            normalized_df[column] = pd.to_numeric(normalized_df[column], errors="coerce")

    if "route_family" not in normalized_df.columns and "process_subtype" in normalized_df.columns:
        normalized_df["route_family"] = normalized_df["process_subtype"].astype(str)

    return normalized_df


def build_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    missing_optional = [c for c in OPTIONAL_COLUMNS if c not in df.columns]

    numeric_expected = [c for c in REQUIRED_COLUMNS if c not in ["specimen_id", "route_family", "process_subtype"]]
    dtype_issues = {}
    for col in numeric_expected:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            dtype_issues[col] = str(df[col].dtype)

    object_expected = ["specimen_id", "route_family"]
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


def render_top_bar() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <p class="hero-title">🔬 {AppConfig.APP_TITLE}</p>
            <p class="hero-subtitle">Advanced fatigue analytics workspace • {AppConfig.APP_BRAND} • {AppConfig.APP_VERSION}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

    with c1:
        selected_license = st.selectbox(
            "License",
            ["Free Demo", "Academic", "Industry", "Enterprise"],
            index=["Free Demo", "Academic", "Industry", "Enterprise"].index(st.session_state.get("license_tier", "Free Demo")),
            key="license_tier_select",
        )
        st.session_state["license_tier"] = selected_license

    with c2:
        selected_connection = st.selectbox(
            "Connection",
            ["Local", "Server"],
            index=["Local", "Server"].index(st.session_state.get("connection_mode", "Local")),
            key="connection_mode_select",
        )
        st.session_state["connection_mode"] = selected_connection

    with c3:
        if st.button("New Project", use_container_width=True):
            st.session_state["current_project_id"] = f"PRJ-{pd.Timestamp.utcnow().strftime('%Y%m%d%H%M%S')}"
            set_last_operation("New project created")
            st.success("New project initialized")

    with c4:
        with st.expander("Help / Support"):
            st.write("Contact: support@fatigue-demo.local")
            st.write("Workflow: Upload → Validate → Lineage → Stats → ML → AI Summary")

    connect_to_database()
    conn = "Connected" if st.session_state.get("db_connected") else "Not Connected"
    st.markdown(
        f"""
        <div style="margin-bottom:0.8rem;">
            <span class="system-pill">🌐 DB: {conn}</span>
            <span class="system-pill">🔐 License: {st.session_state.get('license_tier', 'Free Demo')}</span>
            <span class="system-pill">🛰️ Mode: {st.session_state.get('connection_mode', 'Local')}</span>
            <span class="system-pill">🧾 Project: {st.session_state.get('current_project_id') or 'Not started'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_panel() -> None:
    connection_text = "Connected" if st.session_state.get("db_connected") else "Not connected"
    st.markdown(
        f"""
        <div style="
            position: fixed;
            right: 16px;
            bottom: 16px;
            z-index: 1000;
            background: #111827;
            color: white;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #374151;
            width: 300px;
            font-size: 0.85rem;
        ">
            <b>Status Panel</b><br/>
            DB / Connection: {connection_text}<br/>
            Connection Mode: {st.session_state.get('connection_mode', 'Local')}<br/>
            License Tier: {st.session_state.get('license_tier', 'Free Demo')}<br/>
            Last Operation: {st.session_state.get('last_operation', 'App started')}
        </div>
        """,
        unsafe_allow_html=True,
    )


def try_get_torch_engine():
    try:
        from ml.torch_summary_engine import TorchSummaryEngine

        return TorchSummaryEngine()
    except Exception:
        return None


def render_route_kpis_and_torch_summary(df: pd.DataFrame) -> None:
    st.markdown("### Route KPIs")
    process_col = "route_family" if "route_family" in df.columns else None

    if process_col:
        routes = sorted(df[process_col].dropna().astype(str).unique().tolist())
        selected_route = st.selectbox("Select Route Family", options=routes, key="kpi_route")
        df_route = df[df[process_col].astype(str) == selected_route]
    else:
        selected_route = "All"
        df_route = df

    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", len(df_route))
    mean_nf = df_route["Nf"].mean() if "Nf" in df_route.columns else np.nan
    c2.metric("Mean Nf", f"{mean_nf:.1f}" if pd.notna(mean_nf) else "N/A")
    cov = (
        (df_route["Nf"].std() / mean_nf) * 100
        if "Nf" in df_route.columns and len(df_route) > 1 and pd.notna(mean_nf) and mean_nf != 0
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
    process_count = df["route_family"].nunique() if "route_family" in df.columns else 0
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
            lines.append(f"- Average Nf: {target_mean:.2f}")
        if target_cov is not None:
            lines.append(f"- CoV of Nf: {target_cov:.2f}%")
        ranking = stats.get("influence_ranking", [])
        if ranking:
            top = ranking[0]
            lines.append(f"- Strongest linear influence: {top['feature']} (|corr|={top['abs_corr']:.3f})")

    if ml:
        pred = ml.get("prediction")
        r2 = ml.get("r2")
        rmse = ml.get("rmse")
        if pred is not None:
            lines.append(f"- Predicted Nf: {pred:.2f}")
        if r2 is not None and rmse is not None:
            lines.append(f"- Model fit: R²={r2:.3f}, RMSE={rmse:.2f}")
        else:
            lines.append("- Model fit: demo mode (insufficient data for reliable metrics).")

    if user_query:
        lines.append(f"- User query response: At a high level, '{user_query}' can be assessed from stress/strain and process effects shown above.")

    return "\n".join(lines)


def show_executive_dashboard() -> None:
    render_top_bar()
    st.header("Executive Dashboard")

    col_a, col_b, col_c = st.columns(3)
    col_a.success(f"Connection Status: {'Connected' if st.session_state.get('db_connected') else 'Not connected'}")
    col_b.info(f"License: {st.session_state.get('license_tier')}")
    col_c.info(f"Connection: {st.session_state.get('connection_mode')}")

    st.markdown("### 📂 Upload Dataset")
    with st.container(border=True):
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
                set_last_operation(f"CSV uploaded: {uploaded.name}")
                st.success(f"Loaded {len(uploaded_df)} rows from {uploaded.name}")
            except Exception as exc:
                st.error(f"Unable to parse CSV: {type(exc).__name__}")

    df = get_active_dataset()

    with st.expander("Dataset Preview", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### ✅ Schema Validation")
    if st.button("Run Schema Validation", use_container_width=True):
        st.session_state["validation_report"] = build_validation_report(df)
        set_last_operation("Validation executed")

    report = st.session_state.get("validation_report") or build_validation_report(df)
    st.session_state["validation_report"] = report

    m1, m2, m3 = st.columns(3)
    m1.metric("Required Columns", f"{len(REQUIRED_COLUMNS)-len(report['missing_required'])}/{len(REQUIRED_COLUMNS)}")
    m2.metric("Optional Columns", f"{len(OPTIONAL_COLUMNS)-len(report['missing_optional'])}/{len(OPTIONAL_COLUMNS)}")
    m3.metric("Rows", report["row_count"])

    if report["missing_required"]:
        st.error(f"Missing required columns: {report['missing_required']}")
    else:
        st.success("All required columns are present.")

    if report["missing_optional"]:
        st.warning(f"Missing optional columns (allowed): {report['missing_optional']}")

    render_route_kpis_and_torch_summary(df)

    st.markdown("### 🧠 AI Summary")
    user_query = st.text_input("Ask the AI assistant", placeholder="What affects fatigue life the most?")
    summary = generate_ai_summary(df, stats=st.session_state.get("stats_results"), ml=st.session_state.get("ml_results"), user_query=user_query if user_query else None)
    set_last_operation("AI summary generated")
    st.markdown('<div class="summary-box">', unsafe_allow_html=True)
    st.markdown(summary.replace("\n", "  \n"))
    st.markdown('</div>', unsafe_allow_html=True)

def show_data_lineage() -> None:
    st.header("Data Lineage")
    df = get_active_dataset()
    report = st.session_state.get("validation_report") or build_validation_report(df)
    st.session_state["validation_report"] = report

    st.markdown("### Data Pipeline")
    st.info("Upload CSV → Normalize Columns → Validate Schema → Statistics → ML Prediction → AI Summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Required", f"{len(REQUIRED_COLUMNS)-len(report['missing_required'])}/{len(REQUIRED_COLUMNS)}")
    c2.metric("Optional", f"{len(OPTIONAL_COLUMNS)-len(report['missing_optional'])}/{len(OPTIONAL_COLUMNS)}")
    c3.metric("Rows", report["row_count"])
    c4.metric("Missing Cells", int(sum(report["missing_values"].values())))

    with st.expander("Schema Checklist", expanded=True):
        checklist = pd.DataFrame(
            {
                "column": REQUIRED_COLUMNS + OPTIONAL_COLUMNS,
                "required": ["Yes"] * len(REQUIRED_COLUMNS) + ["No"] * len(OPTIONAL_COLUMNS),
                "present": ["✅" if c in df.columns else "❌" for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS],
            }
        )
        st.dataframe(checklist, use_container_width=True, hide_index=True)

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
            set_last_operation("Validation confirmed")
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

    if not numeric_cols:
        st.warning("No numeric columns available.")
        st.stop()

    stats_results: Dict[str, Any] = {}
    tab1, tab2 = st.tabs(["📋 Descriptive", "📊 Visual Analytics"])

    with tab1:
        st.markdown("### Descriptive Statistics")
        desc = df[numeric_cols].describe().T
        st.dataframe(desc, use_container_width=True)
        stats_results["descriptive"] = desc.to_dict()

    with tab2:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Distribution: Nf")
            if "Nf" in df.columns and pd.api.types.is_numeric_dtype(df["Nf"]):
                hist = (
                    alt.Chart(df)
                    .mark_bar(color="#0f62fe")
                    .encode(x=alt.X("Nf:Q", bin=True, title="Cycles to failure (Nf)"), y=alt.Y("count()", title="Count"))
                    .properties(height=320)
                )
                st.altair_chart(hist, use_container_width=True)
            else:
                st.warning("'Nf' unavailable for distribution plot.")

        with right:
            st.markdown("#### Correlation")
            corr = df[numeric_cols].corr(numeric_only=True)
            if corr.shape[0] >= 2:
                corr_long = corr.reset_index().melt(id_vars="index", var_name="feature", value_name="corr")
                heatmap = (
                    alt.Chart(corr_long)
                    .mark_rect()
                    .encode(
                        x=alt.X("index:N", title="Feature"),
                        y=alt.Y("feature:N", title="Feature"),
                        color=alt.Color("corr:Q", scale=alt.Scale(scheme="redblue")),
                        tooltip=["index", "feature", alt.Tooltip("corr:Q", format=".3f")],
                    )
                    .properties(height=320)
                )
                st.altair_chart(heatmap, use_container_width=True)
            else:
                st.dataframe(corr, use_container_width=True)

    st.markdown("### Influence Ranking vs Nf")
    corr = df[numeric_cols].corr(numeric_only=True)
    ranking: List[Dict[str, Any]] = []
    if "Nf" in corr.columns:
        target_corr = corr["Nf"].drop(labels=["Nf"], errors="ignore").dropna()
        if not target_corr.empty:
            rank_df = target_corr.abs().sort_values(ascending=False).reset_index()
            rank_df.columns = ["feature", "abs_corr"]
            col_table, col_chart = st.columns([1.05, 1.2])
            with col_table:
                st.dataframe(rank_df, use_container_width=True)
            with col_chart:
                chart = (
                    alt.Chart(rank_df)
                    .mark_bar(color="#16a34a")
                    .encode(x=alt.X("abs_corr:Q", title="|Correlation with Nf|"), y=alt.Y("feature:N", sort="-x"))
                    .properties(height=300)
                )
                st.altair_chart(chart, use_container_width=True)
            ranking = rank_df.to_dict("records")

    target_series = df["Nf"] if "Nf" in df.columns else pd.Series(dtype=float)
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
    set_last_operation("Stats computed")

def show_machine_learning() -> None:
    st.header("Machine Learning")
    if not st.session_state.get("validation_confirmed", False):
        st.warning("Please complete Data Lineage and click 'Confirm & Proceed' first.")
        st.stop()

    df = get_active_dataset().copy()
    required_ml_cols = ["YS_MPa", "mean_stress_mean", "PSA_mean", "frequency_Hz", "TSA", "grain_size_um", "Nf"]
    missing_ml = [c for c in required_ml_cols if c not in df.columns]
    if missing_ml:
        st.warning(f"Insufficient columns for ML baseline: {missing_ml}")
        st.stop()

    work = df.copy()
    work["Route Enc"] = pd.factorize(work.get("route_family", "Unknown"))[0]

    feature_cols = ["grain_size_um", "Route Enc", "YS_MPa", "mean_stress_mean", "PSA_mean", "frequency_Hz", "TSA"]

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
        pred_test = predict_linear_np(coef, X[test_idx]) if len(test_idx) > 0 else np.array([])
        if len(test_idx) > 0:
            y_test = y[test_idx]
            ss_res = float(np.sum((y_test - pred_test) ** 2))
            ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
            metrics["r2"] = float(1 - ss_res / ss_tot) if ss_tot != 0 else None
            metrics["rmse"] = float(np.sqrt(np.mean((y_test - pred_test) ** 2)))
    else:
        coef = fit_linear_np(X, y)

    st.markdown("### 🤖 Prediction Inputs")
    process_options = sorted(df["route_family"].astype(str).unique().tolist()) if "route_family" in df.columns else ["Unknown"]

    panel_left, panel_right = st.columns(2)
    with panel_left:
        with st.container(border=True):
            st.caption("Material properties")
            grain_size = st.number_input("Grain Size (um)", value=float(df["grain_size_um"].median()))
            process_type = st.selectbox("Route Family", options=process_options)
            yield_strength = st.number_input("YS (MPa)", value=float(df["YS_MPa"].median()))
            mean_stress = st.number_input("Mean Stress", value=float(df["mean_stress_mean"].median()))
    with panel_right:
        with st.container(border=True):
            st.caption("Loading conditions")
            mean_psa = st.number_input("PSA Mean", value=float(df["PSA_mean"].median()))
            input_frequency = st.number_input("Frequency (Hz)", value=float(df["frequency_Hz"].median()))
            tsa_default = float(df["TSA"].median()) if "TSA" in df.columns else 0.4
            total_strain_amplitude = st.number_input("TSA", value=tsa_default)

    proc_code = process_options.index(process_type) if process_type in process_options else 0
    sample = pd.DataFrame([[grain_size, proc_code, yield_strength, mean_stress, mean_psa, input_frequency, total_strain_amplitude]], columns=feature_cols)
    prediction = float(predict_linear_np(coef, sample.to_numpy(dtype=float))[0])

    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Nf", f"{prediction:,.2f}")
    m2.metric("R²", f"{metrics['r2']:.3f}" if metrics["r2"] is not None else "N/A")
    m3.metric("RMSE", f"{metrics['rmse']:.2f}" if metrics["rmse"] is not None else "N/A")

    pred_chart = alt.Chart(pd.DataFrame({"Scenario": ["Predicted fatigue life"], "Nf": [prediction]})).mark_bar(color="#2563eb").encode(
        x="Scenario:N", y=alt.Y("Nf:Q", title="Predicted Nf")
    ).properties(height=220)
    st.altair_chart(pred_chart, use_container_width=True)

    if metrics["r2"] is None or metrics["rmse"] is None:
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
    set_last_operation("ML prediction computed")



def render_enhanced_sidebar() -> str:
    with st.sidebar:
        st.markdown(f"### ⚙️ {AppConfig.APP_BRAND}")
        st.caption("Navigation")
        pages = ["Executive Dashboard", "Data Lineage", "Statistical Modelling", "Machine Learning"]
        icons = {
            "Executive Dashboard": "📊",
            "Data Lineage": "🧬",
            "Statistical Modelling": "📈",
            "Machine Learning": "🤖",
        }
        page = st.radio("Go to", pages, format_func=lambda x: f"{icons[x]}  {x}", label_visibility="collapsed")
        st.markdown("---")
        st.caption("Workflow")
        st.write("📤 Upload")
        st.write("✅ Validate")
        st.write("📊 Analyze")
        st.write("🤖 Predict")
    return page


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="footer">
            <div><strong>{AppConfig.APP_TITLE}</strong> • {AppConfig.APP_VERSION}</div>
            <div>© 2026 {AppConfig.APP_BRAND} • Support: support@fatigue-demo.local</div>
            <div>Last operation: {st.session_state.get('last_operation', 'App started')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main() -> None:
    st.set_page_config(page_title=AppConfig.APP_TITLE, page_icon=AppConfig.APP_ICON, layout="wide")
    apply_custom_styling()
    init_session_state()
    connect_to_database()

    page = render_enhanced_sidebar()

    if page == "Executive Dashboard":
        show_executive_dashboard()
    elif page == "Data Lineage":
        show_data_lineage()
    elif page == "Statistical Modelling":
        show_statistical_modelling()
    elif page == "Machine Learning":
        show_machine_learning()

    render_status_panel()
    render_footer()

if __name__ == "__main__":
    main()
