"""
Fatigue Data Intelligence Dashboard (Streamlit)

Requirements (install via pip):

    pip install streamlit pandas numpy altair

Optional (for future real integration):
    pip install SQLAlchemy psycopg2-binary scikit-learn

Run with:
    streamlit run streamlit_app.py

Note:
- All DB connections and ML/statistical models are placeholders.
- Replace the marked sections with your real fatigue-DBMS logic.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, Any


# -----------------------------#
# Global / Session Initialization
# -----------------------------#

def init_session_state():
    """Initialize keys in st.session_state to avoid KeyErrors."""
    defaults = {
        "stat_feature_importance": None,
        "ml_feature_importance": None,
        "last_operation": "App started",
        "db_connected": False,   # Simulated DB connection flag
        "current_project_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# -----------------------------#
# Placeholder DB Connection Logic
# -----------------------------#

def connect_to_database() -> bool:
    """
    Simulate a database connection.
    Replace this with real PostgreSQL connection logic using SQLAlchemy/psycopg2.

    Example (future):
        from sqlalchemy import create_engine
        engine = create_engine("postgresql://user:password@host:port/fatigue_dbms_v1")
        conn = engine.connect()
    """
    # Placeholder: always "connected"
    # In real usage, wrap your connection attempt in try/except and return True/False.
    return True


# -----------------------------#
# Dummy Data Generators
# -----------------------------#

def generate_dummy_fatigue_data(n: int = 100) -> pd.DataFrame:
    """Generate a dummy fatigue dataset as fallback when no CSV is uploaded."""
    np.random.seed(42)
    routes = ["T5", "T6A", "T6W", "DCT6", "ECAP90", "ECAP120"]
    df = pd.DataFrame({
        "specimen_id": [f"S{i+1:03d}" for i in range(n)],
        "route_id": np.random.choice(routes, size=n),
        "frequency_hz": np.random.choice([0.1, 0.3, 0.5, 0.7], size=n),
        "tsa_percent": np.full(n, 0.4),
        "temperature_c": np.random.choice([25, 80], size=n),
        "grain_size_um": np.random.uniform(10, 100, size=n),
        "hardness_hv": np.random.uniform(60, 120, size=n),
        "cycles_to_failure": np.random.randint(300, 8000, size=n),
    })
    df["logNf"] = np.log10(df["cycles_to_failure"])
    df["hysteresis_energy"] = np.random.uniform(0.1, 1.0, size=n)
    return df


def generate_dummy_feature_importance(prefix: str) -> pd.DataFrame:
    """Return dummy feature importance for stat or ML perspective."""
    features = ["grain_size_um", "hardness_hv", "frequency_hz", "tsa_percent", "temperature_c"]
    importance = np.random.rand(len(features))
    importance = importance / importance.sum()
    df = pd.DataFrame({
        "feature": features,
        "importance": importance,
        "source": prefix,
    })
    return df.sort_values("importance", ascending=False)


# -----------------------------#
# AI Summary Placeholder
# -----------------------------#

def generate_ai_summary(dummy_stats: Dict[str, Any], dummy_ml_results: Dict[str, Any]) -> str:
    """
    Placeholder AI-generated summary.

    In future, this function can:
    - Use outputs from Statistical Analysis and ML Prediction pages
    - Call an LLM / summarization API
    - Integrate web search results / external context
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


# -----------------------------#
# UI Components: Top Menu Bar
# -----------------------------#

def render_top_bar():
    """
    Render the top horizontal menu bar with:
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
        }
        .top-bar h1 {
            color: white;
            font-size: 1.2rem;
            display: inline-block;
            margin-right: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        cols = st.columns([4, 1, 1])
        with cols[0]:
            st.markdown(
                '<div class="top-bar"><h1>Fatigue Data Intelligence Dashboard</h1></div>',
                unsafe_allow_html=True,
            )
        with cols[1]:
            if st.button("➕ New Project"):
                # Reset / initialize a new project (placeholder)
                st.session_state["current_project_id"] = "project_placeholder"
                st.session_state["last_operation"] = "Started new project (placeholder)"
                st.success("New project started (placeholder). You can now import data.")
        with cols[2]:
            # Help / Contact Support expander
            with st.expander("❓ Help / Contact"):
                st.markdown(
                    """
                    **Support Email (placeholder):** `support@fatigue-dbms.example`

                    **How to use this dashboard (high-level):**
                    1. Go to **Executive Dashboard** and upload fatigue CSV files.
                    2. Explore **Data Lineage** to understand the processing flow.
                    3. Use **Statistical Analysis** to compute descriptive and reliability metrics.
                    4. Use **ML Prediction** to explore model performance and feature importance.
                    """
                )


# -----------------------------#
# UI Components: Status Panel
# -----------------------------#

def render_status_panel():
    """
    Render a small status panel at bottom-right using fixed-position HTML/CSS.
    Shows:
        - DB connection status
        - Last operation
    """
    db_status_color = "#4CAF50" if st.session_state.get("db_connected", False) else "#FF5252"
    db_status_text = "Connected to fatigue_dbms_v1" if st.session_state.get("db_connected", False) else "Not connected"

    last_op = st.session_state.get("last_operation", "No operations yet")

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
        max-width: 260px;
    ">
        <div style="font-weight: 600; margin-bottom: 0.25rem;">Status</div>
        <div style="margin-bottom: 0.25rem;">
            <span style="font-weight: 500;">DB:</span>
            <span style="color: {db_status_color};"> {db_status_text}</span>
        </div>
        <div style="margin-bottom: 0.25rem;">
            <span style="font-weight: 500;">Last operation:</span><br/>
            <span>{last_op}</span>
        </div>
    </div>
    """
    st.markdown(status_html, unsafe_allow_html=True)


# -----------------------------#
# Page 1: Executive Dashboard
# -----------------------------#

def show_executive_dashboard():
    st.subheader("Executive Dashboard")

    # ----- Import Segment -----
    st.markdown("### Import Segment")

    uploaded_files = st.file_uploader(
        "Upload fatigue CSV file(s)",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload processed fatigue test result files (cycles_to_failure, stress/strain features, etc.)."
    )

    if uploaded_files:
        # For simplicity, concatenate all uploaded CSV files
        dfs = []
        for f in uploaded_files:
            df = pd.read_csv(f)
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        st.session_state["last_operation"] = f"Loaded {len(uploaded_files)} CSV file(s)"
    else:
        st.info("No files uploaded. Showing dummy fatigue dataset (placeholder).")
        data = generate_dummy_fatigue_data()
        st.session_state["last_operation"] = "Loaded dummy dataset (no files uploaded)"

    # Store data in session_state for potential cross-page use (optional)
    st.session_state["current_data"] = data

    st.markdown("#### Data Preview")
    st.dataframe(data.head(), width="stretch")

    # ----- Material Type / Alloy -----
    st.markdown("---")
    st.markdown("### Material & Parameters")

    cols = st.columns([1, 1])
    with cols[0]:
        material = st.selectbox(
            "Material Type / Alloy (placeholder)",
            ["Al 6063", "Al 6061", "Steel – placeholder"],
            index=0
        )
    with cols[1]:
        st.markdown("**Selected Material / Alloy:**")
        st.markdown(f"<span style='padding:4px 8px; background-color:#1E88E5; color:white; border-radius:4px;'>{material}</span>",
                    unsafe_allow_html=True)

    st.markdown("#### Example Input Parameters (Testing Conditions - Placeholder)")
    st.markdown(
        """
        - Frequency (Hz): e.g., 0.1, 0.3, 0.5, 0.7  
        - Total Strain Amplitude (TSA %): e.g., 0.4%  
        - Temperature (°C): e.g., 25, 80  
        - Processing Route: e.g., T5, T6A, T6W, DCT, ECAP90, ECAP120  
        """
    )

    st.markdown("#### Example Output Parameters (Fatigue Metrics - Placeholder)")
    st.markdown(
        """
        - Cycles to failure, **Nf**  
        - logNf = log10(Nf)  
        - Hysteresis energy per cycle  
        - Mean stress evolution  
        - Plastic strain amplitude (PSA)  
        """
    )

    # ----- High-level Summary Cards -----
    st.markdown("---")
    st.markdown("### High-level Summary")

    n_records = len(data)
    n_specimens = data["specimen_id"].nunique() if "specimen_id" in data.columns else "N/A"
    n_routes = data["route_id"].nunique() if "route_id" in data.columns else "N/A"
    n_features = data.shape[1]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Records", value=n_records)
    with c2:
        st.metric("Unique Specimens", value=n_specimens)
    with c3:
        st.metric("Unique Routes", value=n_routes)
    with c4:
        st.metric("Feature Count", value=n_features)

    # ----- Feature Importance & Key Drivers -----
    st.markdown("---")
    st.markdown("### Feature Importance & Key Drivers (Placeholder)")

    # Combine statistical and ML feature importance if available
    stat_fi = st.session_state.get("stat_feature_importance")
    ml_fi = st.session_state.get("ml_feature_importance")

    if stat_fi is None:
        stat_fi = generate_dummy_feature_importance("stat")
        st.session_state["stat_feature_importance"] = stat_fi

    if ml_fi is None:
        ml_fi = generate_dummy_feature_importance("ml")
        st.session_state["ml_feature_importance"] = ml_fi

    combined_fi = pd.concat([stat_fi, ml_fi], ignore_index=True)

    st.markdown("#### Combined Feature Importance (Statistical + ML, dummy data)")
    chart = (
        alt.Chart(combined_fi)
        .mark_bar()
        .encode(
            x=alt.X("feature:N", sort="-y", title="Feature"),
            y=alt.Y("importance:Q", title="Relative Importance"),
            color="source:N",
            tooltip=["feature", "importance", "source"]
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

    st.info(
        "Above chart uses dummy data. In future, replace `stat_feature_importance` and "
        "`ml_feature_importance` with real outputs from the Statistical Analysis and ML Prediction pages."
    )

    # ----- AI-enabled Summary Demonstration -----
    st.markdown("---")
    st.markdown("### AI-enabled Summary (Placeholder)")

    dummy_stats = {"highlight": "Mean Nf is highest for ECAP90, followed by DCT6 (dummy)."}
    dummy_ml_results = {"highlight": "ML model ranks grain_size_um and hardness_hv as top drivers (dummy)."}

    summary_text = generate_ai_summary(dummy_stats, dummy_ml_results)

    st.markdown(
        f"""
        <div style="
            border-radius: 8px;
            border: 1px solid #444;
            padding: 1rem;
            background-color: #111827;
        ">
        <strong>AI Summary (placeholder):</strong><br/>
        <pre style="white-space: pre-wrap; font-family: inherit;">{summary_text}</pre>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------#
# Page 2: Data Lineage
# -----------------------------#

def show_data_lineage():
    st.subheader("Data Lineage")

    st.markdown(
        """
        This page provides a conceptual overview of the fatigue data lifecycle in the DBMS.
        """
    )

    st.markdown("### High-level Flow")

    # Diagram-like textual layout using columns and arrows
    stages = [
        "Raw fatigue test files (CSV, loop data)",
        "Data cleansing & ETL (Python scripts)",
        "Normalization & feature engineering",
        "Ingestion into PostgreSQL (fatigue_dbms_v1)",
        "Statistical analysis layer",
        "ML prediction & feature importance layer",
        "Executive dashboard & AI summary layer",
    ]

    for i, stage in enumerate(stages, start=1):
        st.markdown(f"**{i}. {stage}**")

    st.markdown("---")
    st.markdown("### Flowchart (Placeholder)")

    # Simple Graphviz flow to visualize the pipeline
    try:
        import graphviz

        graph_code = """
        digraph G {
            rankdir=LR;
            node [shape=box, style="rounded,filled", fillcolor="#1E293B", fontcolor="white"];

            raw [label="Raw Fatigue Files\\n(CSV / Loop)"];
            etl [label="Cleansing & ETL\\n(Python)"];
            fe [label="Feature Engineering"];
            db [label="PostgreSQL\\n(fatigue_dbms_v1)"];
            stats [label="Statistical Analysis"];
            ml [label="ML Models"];
            dash [label="Executive Dashboard\\n& AI Summary"];

            raw -> etl -> fe -> db -> stats -> ml -> dash;
        }
        """
        st.graphviz_chart(graph_code)
    except Exception:
        st.warning("Graphviz is not installed. Install `graphviz` for flowchart visualization, or skip this step.")

    st.markdown("---")
    st.markdown("### Stage-wise Description (Placeholder)")

    cols = st.columns(3)
    with cols[0]:
        st.markdown("**Raw Data**")
        st.write(
            "Fatigue machine outputs raw CSV and cycle-wise loop files. "
            "These may contain non-physical data after failure and inconsistent units."
        )

    with cols[1]:
        st.markdown("**ETL & Database**")
        st.write(
            "Python scripts perform cleansing, unit normalization, and feature engineering. "
            "The curated data is then ingested into a PostgreSQL schema that mirrors the experimental hierarchy."
        )

    with cols[2]:
        st.markdown("**Analytics & Dashboard**")
        st.write(
            "Statistical and ML layers query the relational database to compute fatigue metrics, "
            "reliability parameters, and feature importance. The executive dashboard then presents these insights."
        )


# -----------------------------#
# Page 3: Statistical Analysis
# -----------------------------#

def show_statistical_analysis():
    st.subheader("Statistical Analysis")

    st.markdown(
        "Select a mathematical model to run on the current dataset. "
        "All results shown here are dummy placeholders; replace with real statistical computations."
    )

    model_choice = st.selectbox(
        "Choose a Statistical / Mathematical Model (placeholder)",
        [
            "Descriptive Statistics",
            "Weibull Analysis",
            "Regression: logNf vs grain_size_um",
            "Regression: logNf vs hardness_hv",
        ],
    )

    # Retrieve dataset (dummy if not set)
    data = st.session_state.get("current_data", generate_dummy_fatigue_data())

    st.markdown("---")

    if model_choice == "Descriptive Statistics":
        st.markdown("### Descriptive Statistics (Placeholder)")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        desc = data[numeric_cols].describe().T
        desc["CoV"] = desc["std"] / desc["mean"]
        st.dataframe(desc, use_container_width=True)
        st.session_state["last_operation"] = "Ran Descriptive Statistics (placeholder)"

        # Dummy feature importance: variance-based
        importance = (data[numeric_cols].var() / data[numeric_cols].var().sum()).reset_index()
        importance.columns = ["feature", "importance"]
        importance["source"] = "stat"
        st.session_state["stat_feature_importance"] = importance

        st.markdown("#### Variance-based Feature Importance (Dummy)")
        st.bar_chart(importance.set_index("feature")["importance"])

    elif model_choice == "Weibull Analysis":
        st.markdown("### Weibull Analysis (Placeholder)")
        st.info(
            "Replace this section with real Weibull fitting: estimate shape (k) and scale (λ) "
            "using fatigue life data and show survival probabilities."
        )

        # Dummy Weibull parameters
        k = 1.2
        lam = 2000
        st.write(f"Shape parameter (k): **{k:.2f}** (dummy)")
        st.write(f"Scale parameter (λ): **{lam:.2f}** cycles (dummy)")

        # Dummy survival curve
        x = np.linspace(100, 5000, 50)
        survival = np.exp(-(x / lam) ** k)
        df_weibull = pd.DataFrame({"cycles": x, "survival_probability": survival})
        line_chart = alt.Chart(df_weibull).mark_line().encode(
            x="cycles",
            y="survival_probability"
        ).properties(title="Weibull Survival Curve (Dummy)", height=300)
        st.altair_chart(line_chart, use_container_width=True)

        st.session_state["last_operation"] = "Ran Weibull Analysis (placeholder)"

    elif model_choice.startswith("Regression: logNf"):
        st.markdown(f"### {model_choice} (Placeholder)")

        target = "logNf"
        # Extract feature name from choice string
        feature_name = model_choice.split(" vs ")[-1]

        if target in data.columns and feature_name in data.columns:
            x = data[feature_name]
            y = data[target]

            # Dummy linear regression using numpy (replace with real statsmodels / sklearn later)
            a, b = np.polyfit(x, y, 1)
            y_pred = a * x + b

            df_reg = pd.DataFrame({
                "x": x,
                "y": y,
                "y_pred": y_pred
            })

            scatter = alt.Chart(df_reg).mark_circle(size=40, opacity=0.5).encode(
                x=alt.X("x", title=feature_name),
                y=alt.Y("y", title=target),
            )

            line = alt.Chart(df_reg).mark_line(color="orange").encode(
                x="x",
                y="y_pred"
            )

            st.altair_chart((scatter + line).properties(height=300), width="stretch")

            st.write(f"Dummy regression line: **{target} ≈ {a:.3f} × {feature_name} + {b:.3f}**")
            st.session_state["last_operation"] = f"Ran Regression ({feature_name}) (placeholder)"

            # Dummy feature importance: single feature with importance 1.0
            importance = pd.DataFrame({
                "feature": [feature_name],
                "importance": [1.0],
                "source": ["stat"]
            })
            st.session_state["stat_feature_importance"] = importance

            st.markdown("#### Regression-based Feature Importance (Dummy)")
            st.bar_chart(importance.set_index("feature")["importance"])
        else:
            st.error(f"Required columns '{target}' or '{feature_name}' not found in current dataset.")

    st.markdown(
        """
        > Note: All computations on this page are placeholders.  
        > Replace them with real statistical analysis scripts and PostgreSQL queries from your fatigue DBMS.
        """
    )


# -----------------------------#
# Page 4: ML Prediction
# -----------------------------#

def show_ml_prediction():
    st.subheader("ML Prediction")

    st.markdown(
        "This page simulates ML model selection and prediction for fatigue life. "
        "All models and metrics are dummy placeholders; replace with real models and metrics later."
    )

    model_choice = st.selectbox(
        "Select ML Model (placeholder)",
        [
            "RandomForest (placeholder)",
            "XGBoost (placeholder)",
            "Linear Regression (placeholder)",
        ],
    )

    st.markdown("---")
    st.markdown("### Model Performance Metrics (Dummy)")

    # Dummy metrics
    np.random.seed(0)
    r2 = np.round(np.random.uniform(0.6, 0.95), 3)
    rmse = np.round(np.random.uniform(0.1, 0.4), 3)
    mae = np.round(np.random.uniform(0.1, 0.3), 3)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("R²", value=r2)
    with m2:
        st.metric("RMSE (logNf)", value=rmse)
    with m3:
        st.metric("MAE (logNf)", value=mae)

    st.session_state["last_operation"] = f"Evaluated ML model: {model_choice} (placeholder)"

    st.markdown("### Feature Importance (ML Perspective - Dummy)")

    ml_fi = generate_dummy_feature_importance("ml")
    st.session_state["ml_feature_importance"] = ml_fi

    chart = alt.Chart(ml_fi).mark_bar().encode(
        x=alt.X("feature:N", sort="-y"),
        y="importance:Q",
        tooltip=["feature", "importance"],
    ).properties(height=300)

    st.altair_chart(chart, width="stretch")

    st.info(
        "Above feature importance values are randomly generated placeholders. "
        "Replace them with real feature_importances_ from your trained ML models "
        "(e.g., RandomForest, XGBoost) or SHAP values."
    )

    st.markdown("---")
    st.markdown("### Predict Fatigue Life (Dummy)")

    st.write("Enter a few input features to simulate a fatigue life prediction:")

    c1, c2, c3 = st.columns(3)
    with c1:
        grain_size = st.number_input("Grain Size (μm)", min_value=1.0, max_value=200.0, value=20.0)
    with c2:
        hardness = st.number_input("Hardness (HV)", min_value=10.0, max_value=300.0, value=90.0)
    with c3:
        frequency = st.number_input("Frequency (Hz)", min_value=0.1, max_value=5.0, value=0.3, step=0.1)

    tsa = st.slider("Total Strain Amplitude (%)", min_value=0.1, max_value=1.0, value=0.4, step=0.1)

    if st.button("Predict (placeholder)"):
        # Dummy prediction logic: purely illustrative
        logNf_pred = 3.0 + 0.001 * (hardness - 80) - 0.002 * (grain_size - 20) - 0.05 * (frequency - 0.3)
        st.success(f"Predicted logNf (dummy): **{logNf_pred:.3f}**")

        # Note: Replace the above with a real ML model prediction:
        #   model.predict([[grain_size, hardness, frequency, tsa, ...]])

    st.markdown(
        """
        > Note: This is a simulation-only page.  
        > Wire this UI to your real ML pipeline: load trained model, perform prediction, and log results.
        """
    )


# -----------------------------#
# Main App
# -----------------------------#

def main():
    st.set_page_config(
        page_title="Fatigue Data Intelligence Dashboard",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()

    # Simulated DB connection (run once at app start)
    if "db_checked" not in st.session_state:
        st.session_state["db_connected"] = connect_to_database()
        st.session_state["db_checked"] = True

    # Top Bar
    render_top_bar()

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        (
            "Executive Dashboard",
            "Data Lineage",
            "Statistical Analysis",
            "ML Prediction",
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Project (placeholder):**")
    st.sidebar.write(st.session_state.get("current_project_id", "None"))

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


if __name__ == "__main__":
    main()