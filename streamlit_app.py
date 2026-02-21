"""
Fatigue Data Intelligence Dashboard (Streamlit) - Enhanced Edition

Requirements (install via pip):
    pip install streamlit pandas numpy altair

Optional (for future real integration):
    pip install SQLAlchemy psycopg2-binary scikit-learn

Run with:
    streamlit run streamlit_app.py

Note:
- All DB connections and ML/statistical models are placeholders.
- Replace the marked sections with your real fatigue-DBMS logic.

ENHANCEMENTS:
- Optimized session state initialization with immutable defaults
- Caching for data generation and feature importance computation
- Robust CSV validation and error handling
- Improved UI responsiveness and consistency
- Better code organization with configuration management
- Enhanced logging for debugging and DBMS integration
- Type hints and docstring improvements
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import logging
from functools import lru_cache
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging for debugging and DBMS integration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ============================
# Configuration & Constants
# ============================

@dataclass
class AppConfig:
    """Centralized configuration for the app."""
    APP_TITLE = "Fatigue Data Intelligence Dashboard"
    APP_ICON = "🧠"
    MATERIAL_TYPES = ["Al 6063", "Al 6061", "Steel – placeholder"]
    PROCESSING_ROUTES = ["T5", "T6A", "T6W", "DCT6", "ECAP90", "ECAP120"]
    FREQUENCY_OPTIONS = [0.1, 0.3, 0.5, 0.7]
    DEFAULT_TSA_PERCENT = 0.4
    TEMPERATURE_OPTIONS = [25, 80]
    FEATURE_COLUMNS = [
        "specimen_id", "route_id", "frequency_hz", "tsa_percent",
        "temperature_c", "grain_size_um", "hardness_hv",
        "cycles_to_failure", "logNf", "hysteresis_energy"
    ]
    DB_NAME = "fatigue_dbms_v1"
    RANDOM_SEED = 42
    DEFAULT_DATASET_SIZE = 100


# ============================
# Session State Management
# ============================

def init_session_state() -> None:
    """Initialize Streamlit session state with immutable defaults."""
    defaults = {
        "stat_feature_importance": None,
        "ml_feature_importance": None,
        "last_operation": "App started",
        "db_connected": False,
        "current_project_id": None,
        "current_data": None,
        "uploaded_file_names": [],
        "data_validation_errors": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    logger.info("Session state initialized")


# ============================
# Database Simulation & Connection
# ============================

def connect_to_database() -> bool:
    """
    Simulate a database connection.

    Replace this with real PostgreSQL connection logic using SQLAlchemy/psycopg2.

    Example (future):
        from sqlalchemy import create_engine
        engine = create_engine("postgresql://user:password@host:port/fatigue_dbms_v1")
        conn = engine.connect()
        return conn is not None
    """
    try:
        # Placeholder: always "connected"
        # In real usage, wrap your connection attempt in try/except and return True/False.
        logger.info(f"Attempting database connection to {AppConfig.DB_NAME}")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# ============================
# Data Generators (Cached)
# ============================

@st.cache_data(ttl=3600, show_spinner=False)
def generate_dummy_fatigue_data(n: int = 100, seed: int = None) -> pd.DataFrame:
    """
    Generate a dummy fatigue dataset as fallback when no CSV is uploaded.

    Args:
        n: Number of records to generate
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: Dummy fatigue dataset with standard columns
    """
    if seed is None:
        seed = AppConfig.RANDOM_SEED

    np.random.seed(seed)

    try:
        df = pd.DataFrame({
            "specimen_id": [f"S{i+1:03d}" for i in range(n)],
            "route_id": np.random.choice(AppConfig.PROCESSING_ROUTES, size=n),
            "frequency_hz": np.random.choice(AppConfig.FREQUENCY_OPTIONS, size=n),
            "tsa_percent": np.full(n, AppConfig.DEFAULT_TSA_PERCENT),
            "temperature_c": np.random.choice(AppConfig.TEMPERATURE_OPTIONS, size=n),
            "grain_size_um": np.random.uniform(10, 100, size=n),
            "hardness_hv": np.random.uniform(60, 120, size=n),
            "cycles_to_failure": np.random.randint(300, 8000, size=n),
        })
        df["logNf"] = np.log10(df["cycles_to_failure"])
        df["hysteresis_energy"] = np.random.uniform(0.1, 1.0, size=n)

        logger.info(f"Generated dummy dataset with {n} records")
        return df
    except Exception as e:
        logger.error(f"Error generating dummy data: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def generate_dummy_feature_importance(prefix: str, seed: int = None) -> pd.DataFrame:
    """
    Return dummy feature importance for statistical or ML perspective.

    Args:
        prefix: Either "stat" or "ml" to label the source
        seed: Random seed for reproducibility

    Returns:
        pd.DataFrame: Feature importance ranking
    """
    if seed is None:
        seed = AppConfig.RANDOM_SEED

    np.random.seed(seed)

    features = ["grain_size_um", "hardness_hv", "frequency_hz", "tsa_percent", "temperature_c"]

    try:
        importance = np.random.rand(len(features))
        importance = importance / importance.sum()

        df = pd.DataFrame({
            "feature": features,
            "importance": importance,
            "source": prefix,
        })

        logger.info(f"Generated {prefix} feature importance")
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error generating feature importance: {e}")
        return pd.DataFrame()


# ============================
# CSV Validation & Parsing
# ============================

def validate_and_parse_csv(file) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Validate and parse an uploaded CSV file with error handling.

    Args:
        file: Streamlit uploaded file object

    Returns:
        Tuple of (DataFrame or None, list of error messages)
    """
    errors = []

    try:
        df = pd.read_csv(file)

        # Basic validation
        if df.empty:
            errors.append(f"{file.name}: File is empty")
            return None, errors

        if len(df) > 100000:
            errors.append(f"{file.name}: Dataset exceeds 100k rows; performance may degrade")
            logger.warning(f"Large dataset detected: {len(df)} rows")

        # Check for critical columns (optional; adjust based on your schema)
        # critical_cols = ["cycles_to_failure"]
        # missing = [c for c in critical_cols if c not in df.columns]
        # if missing:
        #     errors.append(f"{file.name}: Missing critical columns: {missing}")
        #     return None, errors

        logger.info(f"Successfully parsed {file.name}: {len(df)} rows, {len(df.columns)} columns")
        return df, []

    except pd.errors.ParserError as e:
        errors.append(f"{file.name}: CSV parsing error - {str(e)[:100]}")
        logger.error(f"CSV parse error in {file.name}: {e}")
    except Exception as e:
        errors.append(f"{file.name}: Unexpected error - {str(e)[:100]}")
        logger.error(f"Unexpected error parsing {file.name}: {e}")

    return None, errors


# ============================
# AI Summary Generation
# ============================

def generate_ai_summary(dummy_stats: Dict[str, Any], dummy_ml_results: Dict[str, Any]) -> str:
    """
    Placeholder AI-generated summary.

    In future, this function can:
    - Use outputs from Statistical Analysis and ML Prediction pages
    - Call an LLM / summarization API
    - Integrate web search results / external context
    - Query PostgreSQL for real patterns

    Args:
        dummy_stats: Dict with statistical analysis highlights
        dummy_ml_results: Dict with ML analysis highlights

    Returns:
        str: Formatted summary text
    """
    lines = [
        "This is a placeholder AI-generated summary based on statistical and ML results.",
        "",
        f"- Statistical Highlight: {dummy_stats.get('highlight', 'No stats yet, placeholder only.')}",
        f"- ML Model Insight: {dummy_ml_results.get('highlight', 'No ML results yet, placeholder only.')}",
        "",
        "Overall, the current data suggests that certain processing routes and microstructural features "
        "have a strong influence on fatigue life, which can be explored further using the Statistical "
        "Analysis and ML Prediction modules."
    ]
    return "\n".join(lines)


# ============================
# UI Components: Top Menu Bar
# ============================

def render_top_bar() -> None:
    """
    Render the top horizontal menu bar with:
    - Application title
    - New Project button
    - Help / Contact Support
    """
    st.markdown(
        """
        <style>
        .top-bar {
            background-color: #0E1117;
            padding: 0.5rem 1rem 0.5rem 1rem;
            border-bottom: 1px solid #333333;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .top-bar h1 {
            color: white;
            font-size: 1.2rem;
            margin: 0;
        }
        .top-bar-buttons {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        cols = st.columns([4, 1, 1])

        with cols[0]:
            st.markdown(
                '<div class="top-bar"><h1>⚙️ Fatigue Data Intelligence Dashboard</h1></div>',
                unsafe_allow_html=True,
            )

        with cols[1]:
            if st.button("➕ New Project", use_container_width=True):
                st.session_state["current_project_id"] = "project_placeholder"
                st.session_state["last_operation"] = "Started new project (placeholder)"
                st.session_state["current_data"] = None
                st.session_state["uploaded_file_names"] = []
                logger.info("New project initiated via UI")
                st.success("New project started (placeholder). You can now import data.")

        with cols[2]:
            with st.expander("❓ Help / Contact"):
                st.markdown(
                    """
                    <h4 style="text-align:center; margin-top:0;">📞 Contact & Support</h4>
                    <p style="text-align:center;">
                        📧 <strong>Email:</strong> <a href="mailto:service@fdid.in">service@fdid.in</a><br/><br/>
                        💬 <strong>Online Live Support</strong> available during business hours
                    </p>
                    """,
                    unsafe_allow_html=True,
                )


# ============================
# UI Components: Status Panel
# ============================

def render_status_panel() -> None:
    """
    Render a small status panel at bottom-right using fixed-position HTML/CSS.
    Shows:
        - DB connection status
        - Last operation
        - Data record count
    """
    db_status_color = "#4CAF50" if st.session_state.get("db_connected", False) else "#FF5252"
    db_status_text = "Connected to fatigue_dbms_v1" if st.session_state.get("db_connected", False) else "Not connected"

    last_op = st.session_state.get("last_operation", "No operations yet")

    # Count records if data loaded
    record_count = 0
    if st.session_state.get("current_data") is not None:
        record_count = len(st.session_state["current_data"])

    status_html = f"""
    <div style="
        position: fixed;
        bottom: 10px;
        right: 10px;
        background-color: #0E1117;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
        font-size: 0.8rem;
        z-index: 9999;
        max-width: 280px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    ">
        <div style="font-weight: 700; margin-bottom: 0.5rem; color: #64B5F6;">Status</div>
        <div style="margin-bottom: 0.35rem;">
            <span style="font-weight: 600;">DB:</span>
            <span style="color: {db_status_color}; font-weight: 600;"> ● {db_status_text}</span>
        </div>
        <div style="margin-bottom: 0.35rem;">
            <span style="font-weight: 600;">Records:</span>
            <span> {record_count}</span>
        </div>
        <div style="margin-bottom: 0.35rem;">
            <span style="font-weight: 600;">Last op:</span><br/>
            <span style="color: #A0A0A0; font-size: 0.75rem;">{last_op}</span>
        </div>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)


# ============================
# Page 1: Executive Dashboard
# ============================

def show_executive_dashboard() -> None:
    """Executive Dashboard: Import, high-level summary, and AI-driven insights."""
    st.subheader("Executive Dashboard")

    # ===== Import Segment =====
    st.markdown("### 📥 Import Segment")

    uploaded_files = st.file_uploader(
        "Upload fatigue CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload processed fatigue test result files (cycles_to_failure, stress/strain features, etc.)."
    )

    data = None
    validation_errors = []

    if uploaded_files:
        dfs = []
        for f in uploaded_files:
            df, errors = validate_and_parse_csv(f)
            if df is not None:
                dfs.append(df)
            validation_errors.extend(errors)

        if dfs:
            data = pd.concat(dfs, ignore_index=True)
            st.session_state["current_data"] = data
            st.session_state["uploaded_file_names"] = [f.name for f in uploaded_files]
            st.session_state["last_operation"] = f"Loaded {len(uploaded_files)} CSV file(s)"
            logger.info(f"Successfully loaded {len(uploaded_files)} file(s), total {len(data)} rows")

        # Display validation errors if any
        if validation_errors:
            with st.warning("⚠️ Validation Warnings", icon="⚠️"):
                for error in validation_errors:
                    st.write(f"• {error}")
    else:
        st.info("ℹ️ No files uploaded. Showing dummy fatigue dataset (placeholder).")
        data = generate_dummy_fatigue_data()
        st.session_state["current_data"] = data
        st.session_state["last_operation"] = "Loaded dummy dataset (no files uploaded)"
        logger.info("Using placeholder dummy dataset")

    # Retrieve data (fallback to stored or dummy)
    if data is None or data.empty:
        data = st.session_state.get("current_data")
        if data is None or data.empty:
            data = generate_dummy_fatigue_data()
            st.session_state["current_data"] = data

    # ===== Data Preview =====
    st.markdown("#### 📊 Data Preview")

    with st.expander("View full dataset", expanded=False):
        st.dataframe(data, use_container_width=True, height=400)

    # Show summary in compact form
    st.dataframe(data.head(10), use_container_width=True)

    # ===== Material Type / Alloy =====
    st.markdown("---")
    st.markdown("### 🔬 Material & Testing Parameters")

    cols = st.columns([2, 2])
    with cols[0]:
        material = st.selectbox(
            "Material Type / Alloy",
            AppConfig.MATERIAL_TYPES,
            index=0,
            help="Select the base material for this fatigue study"
        )
    with cols[1]:
        st.markdown("**Selected Material:**")
        st.markdown(
            f"<span style='padding:6px 10px; background-color:#1E88E5; color:white; "
            f"border-radius:4px; font-weight:600;'>{material}</span>",
            unsafe_allow_html=True
        )

    # Input/Output Parameters Reference
    cols_params = st.columns(2)
    with cols_params[0]:
        st.markdown("**Input Parameters (Testing Conditions)**")
        st.markdown(
            """
            - **Frequency (Hz):** e.g., 0.1, 0.3, 0.5, 0.7  
            - **Total Strain Amplitude (TSA %):** e.g., 0.4%  
            - **Temperature (°C):** e.g., 25, 80  
            - **Processing Route:** e.g., T5, T6A, T6W, DCT6, ECAP90, ECAP120  
            """
        )
    with cols_params[1]:
        st.markdown("**Output Parameters (Fatigue Metrics)**")
        st.markdown(
            """
            - **Cycles to failure (Nf)**  
            - **logNf = log₁₀(Nf)**  
            - **Hysteresis energy per cycle**  
            - **Mean stress evolution**  
            - **Plastic strain amplitude (PSA)**  
            """
        )

    # ===== High-level Summary Cards =====
    st.markdown("---")
    st.markdown("### 📈 High-level Summary")

    n_records = len(data)
    n_specimens = data["specimen_id"].nunique() if "specimen_id" in data.columns else 0
    n_routes = data["route_id"].nunique() if "route_id" in data.columns else 0
    n_features = data.shape[1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Records", value=n_records, delta=None)
    with c2:
        st.metric("Unique Specimens", value=n_specimens)
    with c3:
        st.metric("Unique Routes", value=n_routes)
    with c4:
        st.metric("Feature Count", value=n_features)

    # ===== Feature Importance & Key Drivers =====
    st.markdown("---")
    st.markdown("### 🎯 Feature Importance & Key Drivers")

    # Combine statistical and ML feature importance
    stat_fi = st.session_state.get("stat_feature_importance")
    ml_fi = st.session_state.get("ml_feature_importance")

    if stat_fi is None:
        stat_fi = generate_dummy_feature_importance("stat")
        st.session_state["stat_feature_importance"] = stat_fi

    if ml_fi is None:
        ml_fi = generate_dummy_feature_importance("ml")
        st.session_state["ml_feature_importance"] = ml_fi

    combined_fi = pd.concat([stat_fi, ml_fi], ignore_index=True)

    # Create Altair chart with proper aggregation
    chart_data = combined_fi.groupby("feature")["importance"].mean().reset_index()
    chart_data = chart_data.sort_values("importance", ascending=False)

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("feature:N", sort="-y", title="Feature Name"),
            y=alt.Y("importance:Q", title="Relative Importance"),
            color=alt.value("#1E88E5"),
            tooltip=["feature", "importance"]
        )
        .properties(height=350, title="Combined Feature Importance (Statistical + ML)")
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "⚠️ Chart uses averaged dummy data. In production, replace with real outputs from "
        "Statistical Analysis and ML Prediction pages."
    )

    # ===== AI-enabled Summary =====
    st.markdown("---")
    st.markdown("### 🤖 AI-enabled Summary")

    dummy_stats = {
        "highlight": "Mean Nf is highest for ECAP90, followed by DCT6 (dummy)."
    }
    dummy_ml_results = {
        "highlight": "ML model ranks grain_size_um and hardness_hv as top drivers (dummy)."
    }

    summary_text = generate_ai_summary(dummy_stats, dummy_ml_results)

    st.markdown(
        f"""
        <div style="
            border-radius: 8px;
            border: 2px solid #4A90D9;
            padding: 1.2rem;
            background-color: rgba(74, 144, 217, 0.08);
            margin: 1rem 0;
        ">
        <strong style="color: #4A90D9;">🤖 AI-Generated Summary (Placeholder):</strong><br/><br/>
        <pre style="white-space: pre-wrap; font-family: 'Segoe UI', sans-serif; font-size:0.92rem; line-height: 1.6; margin:0;">{summary_text}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "**Future Integration:** This summary will leverage real statistical outputs, ML predictions, "
        "web search, and LLM-based generation."
    )


# ============================
# Page 2: Data Lineage
# ============================

def show_data_lineage() -> None:
    """Data Lineage: Visual representation of the fatigue DBMS data flow."""
    st.subheader("Data Lineage")

    st.markdown(
        "This page provides a conceptual overview of the fatigue data lifecycle in the DBMS. "
        "Understanding the lineage is critical for data governance and traceability."
    )

    st.markdown("### 📋 High-level Flow")

    stages = [
        "1. Raw fatigue test files (CSV, loop data)",
        "2. Data cleansing & ETL (Python scripts)",
        "3. Normalization & feature engineering",
        "4. Ingestion into PostgreSQL (fatigue_dbms_v1)",
        "5. Statistical analysis layer",
        "6. ML prediction & feature importance layer",
        "7. Executive dashboard & AI summary layer",
    ]

    for stage in stages:
        st.markdown(f"**{stage}**")

    st.markdown("---")
    st.markdown("### 📊 Flowchart (Placeholder)")

    # Simple Graphviz flow to visualize the pipeline
    try:
        import graphviz

        graph_code = """
        digraph G {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#1E293B", fontcolor="white", fontsize=10];
            edge [color="#666666", penwidth=1.5];

            raw [label="Raw Fatigue Files\\n(CSV / Loop Data)"];
            etl [label="Cleansing & ETL\\n(Python)"];
            fe [label="Feature\\nEngineering"];
            db [label="PostgreSQL\\n(fatigue_dbms_v1)"];
            stats [label="Statistical\\nAnalysis"];
            ml [label="ML Models &\\nFeature Importance"];
            dash [label="Executive Dashboard\\n& AI Summary"];

            raw -> etl -> fe -> db -> stats -> ml -> dash;
        }
        """
        st.graphviz_chart(graph_code, use_container_width=True)
    except ImportError:
        st.warning(
            "📦 **Graphviz not installed.** Install via `pip install graphviz` for flowchart visualization. "
            "You can skip this step and proceed."
        )

    st.markdown("---")
    st.markdown("### 🔍 Stage-wise Description")

    cols = st.columns(3)

    with cols[0]:
        st.markdown("#### 📥 Raw Data")
        st.write(
            "Fatigue machine outputs raw CSV and cycle-wise loop files. "
            "These may contain non-physical data after failure and inconsistent units."
        )

    with cols[1]:
        st.markdown("#### 🔧 ETL & Database")
        st.write(
            "Python scripts perform cleansing, unit normalization, and feature engineering. "
            "The curated data is ingested into a PostgreSQL schema mirroring the experimental hierarchy."
        )

    with cols[2]:
        st.markdown("#### 📊 Analytics & Dashboard")
        st.write(
            "Statistical and ML layers query the relational database to compute fatigue metrics, "
            "reliability parameters, and feature importance. The dashboard presents these insights."
        )

    st.markdown("---")
    st.markdown("### 🛠️ Configuration (Placeholder)")

    st.info(
        "**Future DB Configuration:**\n\n"
        "```sql\n"
        "-- PostgreSQL fatigue_dbms_v1 schema (placeholder)\n"
        "CREATE TABLE specimens (...)\n"
        "CREATE TABLE fatigue_cycles (...)\n"
        "CREATE TABLE microstructure (...)\n"
        "CREATE TABLE test_conditions (...)\n"
        "```"
    )

    # ===== Data Quality Checks =====
    st.markdown("---")
    st.markdown("### 🧪 Data Quality Checks")

    data = st.session_state.get("current_data")
    if data is None or data.empty:
        st.info("No dataset loaded. Upload data via the Executive Dashboard to run quality checks.")
    else:
        EXPECTED_TYPES = {
            "specimen_id": "object",
            "route_id": "object",
            "frequency_hz": "float64",
            "tsa_percent": "float64",
            "temperature_c": "float64",
            "grain_size_um": "float64",
            "hardness_hv": "float64",
            "cycles_to_failure": "int64",
            "logNf": "float64",
            "hysteresis_energy": "float64",
        }

        issues_found = False

        # --- Missing Data Check ---
        missing_summary = data.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        if not missing_cols.empty:
            issues_found = True
            for col, count in missing_cols.items():
                pct = (count / len(data)) * 100
                row_indices = data[data[col].isnull()].index.tolist()
                st.warning(
                    f"⚠️ **Missing Data Detected** — Column `{col}`: "
                    f"**{count} missing value(s)** ({pct:.1f}% of rows)\n\n"
                    f"📍 Affected row indices: `{row_indices[:10]}{'...' if len(row_indices) > 10 else ''}`"
                )

        # --- Data Type Mismatch Check ---
        for col, expected in EXPECTED_TYPES.items():
            if col not in data.columns:
                issues_found = True
                st.warning(
                    f"⚠️ **Column Missing from Dataset** — Expected column `{col}` (type: `{expected}`) not found."
                )
            else:
                actual_dtype = str(data[col].dtype)
                # Allow numeric flexibility (int32/int64, float32/float64)
                type_ok = (
                    actual_dtype == expected
                    or (expected == "float64" and "float" in actual_dtype)
                    or (expected == "int64" and "int" in actual_dtype)
                    or (expected == "object" and actual_dtype == "object")
                )
                if not type_ok:
                    issues_found = True
                    type_label = "text/string" if expected == "object" else ("integer" if "int" in expected else "decimal/precision")
                    st.warning(
                        f"⚠️ **Data Type Mismatch** — Column `{col}`: "
                        f"Expected **{type_label}** (`{expected}`), "
                        f"but found `{actual_dtype}`."
                    )

        if not issues_found:
            st.success("✅ No data quality issues detected. All columns present and types are valid.")
        else:
            st.error("❌ Data quality issues found above. Please review and correct before analysis.")


# ============================
# Page 3: Statistical Analysis
# ============================

def show_statistical_analysis() -> None:
    """Statistical Analysis: Model selection and computation."""
    st.subheader("Statistical Analysis")

    st.markdown(
        "Select a mathematical model to run on the current dataset. "
        "All results shown here are dummy placeholders; replace with real statistical computations."
    )

    model_choice = st.selectbox(
        "Choose a Statistical / Mathematical Model",
        [
            "Descriptive Statistics",
            "Weibull Analysis",
            "Regression: logNf vs grain_size_um",
            "Regression: logNf vs hardness_hv",
        ],
        help="Select a statistical model to apply to the loaded dataset"
    )

    # Retrieve dataset
    data = st.session_state.get("current_data", generate_dummy_fatigue_data())

    if data is None or data.empty:
        data = generate_dummy_fatigue_data()
        st.session_state["current_data"] = data

    st.markdown("---")

    if model_choice == "Descriptive Statistics":
        show_descriptive_statistics(data)

    elif model_choice == "Weibull Analysis":
        show_weibull_analysis(data)

    elif model_choice.startswith("Regression: logNf"):
        feature_name = model_choice.split(" vs ")[-1]
        show_regression_analysis(data, feature_name)

    st.markdown("---")
    st.markdown("### 📝 Integration Notes")
    st.info(
        "All computations on this page are placeholders.\n\n"
        "**Replace with:**\n"
        "- Real SQL queries from PostgreSQL fatigue_dbms_v1\n"
        "- Your existing Phase 1–3 analysis scripts\n"
        "- Store feature importance in `st.session_state['stat_feature_importance']`"
    )


def show_descriptive_statistics(data: pd.DataFrame) -> None:
    """Compute and display descriptive statistics."""
    st.markdown("### 📊 Descriptive Statistics")

    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            st.error("No numeric columns found in dataset")
            return

        desc = data[numeric_cols].describe().T
        desc["CoV"] = desc["std"] / desc["mean"]

        st.dataframe(desc, use_container_width=True)
        st.session_state["last_operation"] = "Ran Descriptive Statistics (placeholder)"
        logger.info("Descriptive statistics computed")

        # Variance-based feature importance
        importance = (data[numeric_cols].var() / data[numeric_cols].var().sum()).reset_index()
        importance.columns = ["feature", "importance"]
        importance["source"] = "stat"
        st.session_state["stat_feature_importance"] = importance

        st.markdown("#### 📈 Variance-based Feature Importance (Dummy)")
        chart = (
            alt.Chart(importance.sort_values("importance", ascending=False))
            .mark_bar()
            .encode(
                x=alt.X("feature:N", sort="-y"),
                y="importance:Q",
                color=alt.value("#FF9800")
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error computing descriptive statistics: {str(e)[:100]}")
        logger.error(f"Descriptive statistics error: {e}")


def show_weibull_analysis(data: pd.DataFrame) -> None:
    """Compute and display Weibull analysis."""
    st.markdown("### 📊 Weibull Analysis")

    st.info(
        "**Future Implementation:** Replace this with real Weibull fitting: estimate shape (k) and scale (λ) "
        "using fatigue life data and show survival probabilities."
    )

    try:
        # Dummy Weibull parameters
        k = 1.2
        lam = 2000
        st.write(f"Shape parameter (k): **{k:.2f}** (dummy)")
        st.write(f"Scale parameter (λ): **{lam:.2f}** cycles (dummy)")

        # Dummy survival curve
        x = np.linspace(100, 5000, 50)
        survival = np.exp(-(x / lam) ** k)
        df_weibull = pd.DataFrame({"cycles": x, "survival_probability": survival})

        line_chart = alt.Chart(df_weibull).mark_line(point=True).encode(
            x=alt.X("cycles:Q", title="Cycles"),
            y=alt.Y("survival_probability:Q", title="Survival Probability"),
            color=alt.value("#4CAF50")
        ).properties(title="Weibull Survival Curve (Dummy)", height=350).interactive()

        st.altair_chart(line_chart, use_container_width=True)

        st.session_state["last_operation"] = "Ran Weibull Analysis (placeholder)"
        logger.info("Weibull analysis computed")

    except Exception as e:
        st.error(f"Error in Weibull analysis: {str(e)[:100]}")
        logger.error(f"Weibull analysis error: {e}")


def show_regression_analysis(data: pd.DataFrame, feature_name: str) -> None:
    """Compute and display regression analysis."""
    st.markdown(f"### 📊 Regression: logNf vs {feature_name}")

    target = "logNf"

    try:
        if target not in data.columns:
            st.error(f"Target column '{target}' not found in dataset")
            return

        if feature_name not in data.columns:
            st.error(f"Feature column '{feature_name}' not found in dataset")
            return

        x = data[feature_name].dropna()
        y = data.loc[x.index, target].dropna()

        if len(x) < 2:
            st.error("Insufficient data for regression")
            return

        # Dummy linear regression
        a, b = np.polyfit(x, y, 1)
        y_pred = a * x + b

        # R² calculation
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        df_reg = pd.DataFrame({
            "x": x.values,
            "y": y.values,
            "y_pred": y_pred.values
        })

        scatter = alt.Chart(df_reg).mark_circle(size=40, opacity=0.6).encode(
            x=alt.X("x:Q", title=feature_name),
            y=alt.Y("y:Q", title=target),
            color=alt.value("#2196F3")
        )

        line = alt.Chart(df_reg).mark_line(color="#FF9800", size=3).encode(
            x="x:Q",
            y="y_pred:Q"
        )

        chart = (scatter + line).properties(
            height=350,
            title=f"Regression: {target} ≈ {a:.4f} × {feature_name} + {b:.4f} (Dummy)"
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # Summary metrics
        cols = st.columns(3)
        with cols[0]:
            st.metric("R² Score", value=f"{r2:.4f}")
        with cols[1]:
            st.metric("Slope (a)", value=f"{a:.6f}")
        with cols[2]:
            st.metric("Intercept (b)", value=f"{b:.4f}")

        st.session_state["last_operation"] = f"Ran Regression ({feature_name}) (placeholder)"
        logger.info(f"Regression analysis computed: {feature_name} vs {target}")

        # Dummy feature importance: single feature
        importance = pd.DataFrame({
            "feature": [feature_name],
            "importance": [abs(r2)],
            "source": ["stat"]
        })
        st.session_state["stat_feature_importance"] = importance

        st.markdown("#### 📈 Regression-based Feature Importance (Dummy)")
        bar_chart = alt.Chart(importance).mark_bar().encode(
            x=alt.X("feature:N"),
            y="importance:Q",
            color=alt.value("#FF9800")
        ).properties(height=250)
        st.altair_chart(bar_chart, use_container_width=True)

    except Exception as e:
        st.error(f"Error in regression analysis: {str(e)[:100]}")
        logger.error(f"Regression analysis error: {e}")


# ============================
# Page 4: ML Prediction
# ============================

def show_ml_prediction() -> None:
    """ML Prediction: Model selection, evaluation, and fatigue life prediction."""
    st.subheader("ML Prediction")

    st.markdown(
        "This page simulates ML model selection and prediction for fatigue life. "
        "All models and metrics are dummy placeholders; replace with real models and metrics later."
    )

    model_choice = st.selectbox(
        "Select ML Model",
        [
            "RandomForest (placeholder)",
            "XGBoost (placeholder)",
            "Linear Regression (placeholder)",
        ],
        help="Select a machine learning model for fatigue life prediction"
    )

    st.markdown("---")
    st.markdown("### 📊 Model Performance Metrics (Dummy)")

    # Dummy metrics with consistency
    np.random.seed(AppConfig.RANDOM_SEED)
    r2 = np.round(np.random.uniform(0.6, 0.95), 3)
    rmse = np.round(np.random.uniform(0.1, 0.4), 3)
    mae = np.round(np.random.uniform(0.1, 0.3), 3)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("R² Score", value=r2, delta=None)
    with m2:
        st.metric("RMSE (logNf)", value=rmse)
    with m3:
        st.metric("MAE (logNf)", value=mae)

    st.session_state["last_operation"] = f"Evaluated ML model: {model_choice} (placeholder)"
    logger.info(f"ML model evaluated: {model_choice}")

    st.markdown("### 🎯 Feature Importance (ML Perspective - Dummy)")

    ml_fi = generate_dummy_feature_importance("ml")
    st.session_state["ml_feature_importance"] = ml_fi

    chart = alt.Chart(ml_fi.head(5)).mark_bar().encode(
        x=alt.X("feature:N", sort="-y", title="Feature"),
        y=alt.Y("importance:Q", title="Importance Score"),
        color=alt.value("#4CAF50"),
        tooltip=["feature", "importance"]
    ).properties(height=300, title="Top 5 Features by Importance (ML)").interactive()

    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "⚠️ Above feature importance values are randomly generated placeholders. "
        "Replace with real feature_importances_ from trained ML models (RandomForest, XGBoost) or SHAP values."
    )

    st.markdown("---")
    st.markdown("### 🔮 Predict Fatigue Life (Dummy)")

    st.write("Enter input features to simulate a fatigue life prediction:")

    c1, c2, c3 = st.columns(3)
    with c1:
        grain_size = st.number_input(
            "Grain Size (μm)",
            min_value=1.0, max_value=200.0, value=20.0, step=1.0,
            help="Grain size in micrometers"
        )
    with c2:
        hardness = st.number_input(
            "Hardness (HV)",
            min_value=10.0, max_value=300.0, value=90.0, step=5.0,
            help="Hardness in Vickers scale"
        )
    with c3:
        frequency = st.number_input(
            "Frequency (Hz)",
            min_value=0.1, max_value=5.0, value=0.3, step=0.1,
            help="Test frequency in Hertz"
        )

    tsa = st.slider(
        "Total Strain Amplitude (%)",
        min_value=0.1, max_value=1.0, value=0.4, step=0.1,
        help="Total strain amplitude as percentage"
    )

    if st.button("🔍 Predict (Placeholder)", use_container_width=True):
        try:
            # Dummy prediction logic: purely illustrative
            logNf_pred = (
                3.0
                + 0.001 * (hardness - 80)
                - 0.002 * (grain_size - 20)
                - 0.05 * (frequency - 0.3)
                - 0.5 * (tsa - 0.4)
            )
            Nf_pred = 10 ** logNf_pred

            st.success(
                f"✅ **Predicted Fatigue Life (Dummy):**\n\n"
                f"- logNf: **{logNf_pred:.3f}**\n"
                f"- Nf (cycles): **{Nf_pred:.0f}**"
            )

            logger.info(
                f"Prediction made: Nf={Nf_pred:.0f}, "
                f"grain_size={grain_size}, hardness={hardness}, freq={frequency}, tsa={tsa}"
            )

            # Note: Replace the above with a real ML model prediction:
            #   model.predict([[grain_size, hardness, frequency, tsa, ...]])

        except Exception as e:
            st.error(f"Prediction error: {str(e)[:100]}")
            logger.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.markdown("### 📝 Integration Notes")
    st.info(
        "This is a simulation-only page.\n\n"
        "**Wire this UI to your real ML pipeline:**\n"
        "- Load trained model from disk or PostgreSQL\n"
        "- Perform prediction using model.predict(...)\n"
        "- Log results and predictions to the DBMS"
    )


# ============================
# Main App
# ============================

def main() -> None:
    """Main application entry point."""
    st.set_page_config(
        page_title=AppConfig.APP_TITLE,
        page_icon=AppConfig.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    init_session_state()

    # Simulated DB connection (run once at app start)
    if "db_checked" not in st.session_state:
        st.session_state["db_connected"] = connect_to_database()
        st.session_state["db_checked"] = True

    # Top expandable sections
    with st.expander("🔌 Connect to Server"):
        server_type = st.radio(
            "Select connection type:",
            ["Internal Server", "Cloud Server"],
            horizontal=True,
        )
        if server_type == "Internal Server":
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("Host / IP Address", placeholder="e.g. 192.168.1.100")
                st.text_input("Database Name", value="fatigue_dbms_v1")
            with col2:
                st.text_input("Port", value="5432")
                st.text_input("Username", placeholder="db_user")
            st.text_input("Password", type="password", placeholder="••••••••")
            if st.button("🔗 Connect to Internal Server", use_container_width=True):
                st.success("✅ Internal server connection initiated (placeholder). Replace with real connection logic.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("Cloud Provider", ["AWS RDS", "Google Cloud SQL", "Azure SQL", "Supabase", "Other"])
                st.text_input("Connection String / Endpoint", placeholder="e.g. mydb.cluster.amazonaws.com")
            with col2:
                st.text_input("Database Name", value="fatigue_dbms_v1")
                st.text_input("Username", placeholder="cloud_user")
            st.text_input("Password / API Key", type="password", placeholder="••••••••")
            if st.button("☁️ Connect to Cloud Server", use_container_width=True):
                st.success("✅ Cloud server connection initiated (placeholder). Replace with real cloud connection logic.")

    with st.expander("📄 License Information"):
        st.markdown(
            """
            <div style="padding: 0.5rem 0;">
                <h4 style="margin-top:0;">🔐 Software License</h4>
                <p><strong>Product:</strong> Fatigue Data Intelligence Dashboard (FDID)</p>
                <p><strong>License Type:</strong> Proprietary — Internal Use Only</p>
                <p><strong>License Holder:</strong> <em>Your Organization Name</em></p>
                <p><strong>Version:</strong> 1.0.0 (Enhanced Edition)</p>
                <p><strong>Valid Until:</strong> <em>Refer to your license agreement</em></p>
                <p><strong>Support Contact:</strong> <a href="mailto:service@fdid.in">service@fdid.in</a></p>
                <hr/>
                <p style="font-size:0.85rem; color: gray;">
                    This software is licensed, not sold. Unauthorized reproduction, distribution, or modification 
                    is strictly prohibited. All rights reserved © 2025 FDID.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Top Bar
    render_top_bar()

    # Sidebar Navigation
    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Go to",
        (
            "Executive Dashboard",
            "Data Lineage",
            "Statistical Analysis",
            "ML Prediction",
        ),
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📦 Current Project (Placeholder):**")
    st.sidebar.write(st.session_state.get("current_project_id") or "None")

    st.sidebar.markdown("**📁 Uploaded Files:**")
    uploaded_files = st.session_state.get("uploaded_file_names", [])
    if uploaded_files:
        for fname in uploaded_files:
            st.sidebar.write(f"✓ {fname}")
    else:
        st.sidebar.write("*No files uploaded*")

    # Main Content Routing
    if page == "Executive Dashboard":
        show_executive_dashboard()
    elif page == "Data Lineage":
        show_data_lineage()
    elif page == "Statistical Analysis":
        show_statistical_analysis()
    elif page == "ML Prediction":
        show_ml_prediction()

    # Status Panel at bottom-right
    render_status_panel()

    logger.info(f"Page rendered: {page}")


if __name__ == "__main__":
    main()