# Integration Checklist: From Placeholder to Production

## Phase 0: Validation & Testing (Before Any Integration)

- [ ] Run `streamlit run streamlit_app_enhanced.py` locally
- [ ] Verify all four pages load without errors
- [ ] Test CSV upload with sample fatigue data
- [ ] Confirm dummy data generation works
- [ ] Check status panel displays correctly
- [ ] Verify all charts render (if graphviz unavailable, should gracefully degrade)

---

## Phase 1: Database Connection

### Task 1.1: Replace `connect_to_database()`

**Current Code:**
```python
def connect_to_database() -> bool:
    """Simulate a database connection."""
    logger.info(f"Attempting database connection to {AppConfig.DB_NAME}")
    return True
```

**Integration Template:**
```python
def connect_to_database() -> bool:
    """Connect to PostgreSQL fatigue_dbms_v1."""
    try:
        from sqlalchemy import create_engine, text
        
        # Replace with your actual credentials
        db_url = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        engine = create_engine(db_url, pool_pre_ping=True)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logger.info(f"✓ Connected to {AppConfig.DB_NAME}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Connection failed: {e}")
        return False
```

**Action Items:**
- [ ] Configure environment variables for `DB_USER`, `DB_PASS`, `DB_HOST`, `DB_PORT`
- [ ] Create `AppConfig.DB_URL` property
- [ ] Add connection pooling with `pool_size=5, max_overflow=10`
- [ ] Test connection on app startup
- [ ] Add retry logic with exponential backoff

---

## Phase 2: Data Loading

### Task 2.1: Add Real Data Query Function

**New Function Location:** After `connect_to_database()`

```python
@st.cache_data(ttl=3600)
def load_fatigue_data_from_db(project_id: Optional[str] = None, 
                              limit: int = 10000) -> pd.DataFrame:
    """
    Load fatigue data from PostgreSQL backend.
    
    Args:
        project_id: Filter by project (if None, load recent data)
        limit: Maximum rows to return
        
    Returns:
        pd.DataFrame with fatigue test results
    """
    try:
        from sqlalchemy import create_engine, text
        
        engine = create_engine(AppConfig.DB_URL)
        
        if project_id:
            query = f"""
                SELECT * FROM fatigue_cycles 
                WHERE project_id = '{project_id}'
                LIMIT {limit}
            """
        else:
            query = f"""
                SELECT * FROM fatigue_cycles 
                ORDER BY created_at DESC 
                LIMIT {limit}
            """
        
        df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(df)} records from PostgreSQL")
        return df
    
    except Exception as e:
        logger.error(f"Failed to load from DB: {e}")
        return pd.DataFrame()
```

**Action Items:**
- [ ] Inspect your PostgreSQL schema (table name, column names)
- [ ] Update query to match your actual tables
- [ ] Add filters (date range, material type, processing route)
- [ ] Implement pagination for large datasets
- [ ] Cache with appropriate TTL (3600 = 1 hour)

### Task 2.2: Modify Executive Dashboard Data Loading

**Current Code in `show_executive_dashboard()`:**
```python
if uploaded_files:
    # Parse CSVs...
else:
    st.info("ℹ️ No files uploaded. Showing dummy fatigue dataset (placeholder).")
    data = generate_dummy_fatigue_data()
```

**Replace With:**
```python
if uploaded_files:
    # Keep CSV upload capability
    dfs = []
    for f in uploaded_files:
        df, errors = validate_and_parse_csv(f)
        if df is not None:
            dfs.append(df)
        validation_errors.extend(errors)
    
    if dfs:
        data = pd.concat(dfs, ignore_index=True)
        # Also optionally save to PostgreSQL
        save_to_db(data, project_id=st.session_state["current_project_id"])

else:
    # Try PostgreSQL first, fallback to dummy
    if st.session_state.get("db_connected", False):
        st.info("ℹ️ Loading from PostgreSQL fatigue_dbms_v1...")
        data = load_fatigue_data_from_db()
    else:
        st.warning("⚠️ Database not available. Using dummy dataset.")
        data = generate_dummy_fatigue_data()
```

**Action Items:**
- [ ] Implement `save_to_db(df, project_id)` to persist uploaded data
- [ ] Add project selector dropdown to load historical projects
- [ ] Implement data refresh button (`st.button("🔄 Refresh from DB")`)

---

## Phase 3: Statistical Analysis Integration

### Task 3.1: Replace Dummy Descriptive Statistics

**Current Code:**
```python
def show_descriptive_statistics(data: pd.DataFrame) -> None:
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    desc = data[numeric_cols].describe().T
    desc["CoV"] = desc["std"] / desc["mean"]
    st.dataframe(desc, use_container_width=True)
```

**Integrated Version:**
```python
def show_descriptive_statistics(data: pd.DataFrame) -> None:
    """Compute real statistical metrics from fatigue data."""
    try:
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Standard descriptive stats
        desc = data[numeric_cols].describe().T
        desc["CoV"] = desc["std"] / desc["mean"]
        
        # Add fatigue-specific metrics
        if "cycles_to_failure" in data.columns:
            cycles = data["cycles_to_failure"]
            desc.loc["cycles_to_failure", "Median"] = cycles.median()
            desc.loc["cycles_to_failure", "Q1"] = cycles.quantile(0.25)
            desc.loc["cycles_to_failure", "Q3"] = cycles.quantile(0.75)
        
        st.dataframe(desc, use_container_width=True)
        
        # Store in session for dashboard
        importance = (data[numeric_cols].var() / data[numeric_cols].var().sum()).reset_index()
        importance.columns = ["feature", "importance"]
        importance["source"] = "stat"
        st.session_state["stat_feature_importance"] = importance
        
        logger.info("Descriptive statistics computed and cached")
    
    except Exception as e:
        st.error(f"Statistical analysis failed: {e}")
        logger.error(f"Stats error: {e}")
```

**Action Items:**
- [ ] Add fatigue-specific metrics (percentiles, distribution shape)
- [ ] Compute confidence intervals for mean Nf
- [ ] Add by-group statistics (by route, temperature, etc.)
- [ ] Export results to CSV

### Task 3.2: Implement Real Weibull Analysis

**Placeholder Code:**
```python
def show_weibull_analysis(data: pd.DataFrame) -> None:
    """Dummy Weibull fitting."""
    k = 1.2
    lam = 2000
    # ...
```

**Real Implementation:**
```python
def show_weibull_analysis(data: pd.DataFrame) -> None:
    """Fit Weibull distribution to fatigue life data."""
    try:
        from scipy.stats import weibull_min
        
        cycles = data["cycles_to_failure"].dropna()
        
        # Fit Weibull
        k, _, lam = weibull_min.fit(cycles)
        
        st.write(f"**Shape parameter (k):** {k:.4f}")
        st.write(f"**Scale parameter (λ):** {lam:.2f} cycles")
        
        # Survival curve
        x = np.linspace(cycles.min(), cycles.max(), 100)
        survival = np.exp(-(x / lam) ** k)
        
        df_weibull = pd.DataFrame({
            "cycles": x,
            "survival_probability": survival,
            "failure_probability": 1 - survival
        })
        
        # Plot
        chart = alt.Chart(df_weibull).mark_line().encode(
            x="cycles:Q",
            y=alt.Y("survival_probability:Q", scale=alt.Scale(domain=[0, 1]))
        ).properties(title=f"Weibull Fit (k={k:.3f}, λ={lam:.0f})")
        
        st.altair_chart(chart, use_container_width=True)
        
        # Reliability at key cycles
        key_cycles = [cycles.quantile(q) for q in [0.1, 0.5, 0.9]]
        for nc in key_cycles:
            R_nc = np.exp(-(nc / lam) ** k)
            st.write(f"Reliability at {nc:.0f} cycles: {R_nc:.3f}")
        
        logger.info(f"Weibull fit: k={k:.4f}, λ={lam:.2f}")
    
    except Exception as e:
        st.error(f"Weibull analysis failed: {e}")
        logger.error(f"Weibull error: {e}")
```

**Action Items:**
- [ ] Install scipy: `pip install scipy`
- [ ] Test on real fatigue data (verify k, λ make physical sense)
- [ ] Add confidence intervals for parameters
- [ ] Compare different processing routes

### Task 3.3: Link to Your Phase 1–3 Scripts

**In `show_statistical_analysis()`:**
```python
# At the top of function
import sys
sys.path.insert(0, "/path/to/your/phase1_scripts")

from phase1_processing import compute_cyclic_hardening
from phase2_microstructure import analyze_grain_evolution
from phase3_reliability import compute_reliability_metrics

# Then use in branches:
if model_choice == "Cyclic Hardening Analysis":
    results = compute_cyclic_hardening(data)
    st.dataframe(results)
```

**Action Items:**
- [ ] List all analysis functions from Phase 1, 2, 3
- [ ] Add new dropdowns for each phase's analyses
- [ ] Ensure data schema compatibility
- [ ] Test integration on sample datasets

---

## Phase 4: ML Model Integration

### Task 4.1: Load Pre-trained Model

**Current Code:**
```python
def show_ml_prediction() -> None:
    # Dummy prediction logic
    logNf_pred = 3.0 + 0.001 * (hardness - 80) - ...
```

**Integrated Version:**
```python
@st.cache_resource
def load_ml_model(model_name: str = "rf_fatigue_life"):
    """Load pre-trained fatigue prediction model."""
    try:
        import joblib
        model = joblib.load(f"models/{model_name}.pkl")
        logger.info(f"Loaded model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None

def show_ml_prediction() -> None:
    """ML Prediction with real model."""
    st.subheader("ML Prediction")
    
    # Load model once (cached)
    model = load_ml_model()
    
    if model is None:
        st.error("ML model not available. Using placeholder.")
        use_placeholder = True
    else:
        use_placeholder = False
    
    # Input features
    feature_inputs = {
        "grain_size_um": st.number_input("Grain Size (μm)", value=20.0),
        "hardness_hv": st.number_input("Hardness (HV)", value=90.0),
        "frequency_hz": st.number_input("Frequency (Hz)", value=0.3),
        "tsa_percent": st.slider("TSA (%)", 0.1, 1.0, 0.4),
    }
    
    if st.button("🔍 Predict"):
        if use_placeholder:
            # Dummy
            logNf_pred = 3.0 + 0.001 * (feature_inputs["hardness_hv"] - 80)
        else:
            # Real model
            X = np.array([list(feature_inputs.values())])
            logNf_pred = model.predict(X)[0]
        
        Nf_pred = 10 ** logNf_pred
        st.success(f"Predicted Nf: **{Nf_pred:.0f}** cycles (logNf={logNf_pred:.3f})")
        
        # Log prediction
        log_prediction(feature_inputs, logNf_pred)
```

**Action Items:**
- [ ] Train and serialize model: `joblib.dump(model, "models/rf_fatigue_life.pkl")`
- [ ] Document feature order and normalization
- [ ] Add model versioning (e.g., `rf_fatigue_life_v2.pkl`)
- [ ] Implement A/B testing for multiple models

### Task 4.2: Feature Importance from Real Model

**Current Code:**
```python
def show_ml_prediction() -> None:
    ml_fi = generate_dummy_feature_importance("ml")
    st.session_state["ml_feature_importance"] = ml_fi
```

**Integrated Version:**
```python
def extract_feature_importance(model) -> pd.DataFrame:
    """Extract feature importance from trained model."""
    try:
        if hasattr(model, "feature_importances_"):
            # Tree-based models (RandomForest, XGBoost)
            importance = model.feature_importances_
            feature_names = model.feature_names_  # or get from training data
        
        elif hasattr(model, "coef_"):
            # Linear models
            importance = np.abs(model.coef_).flatten()
            feature_names = model.feature_names_
        
        else:
            # Use SHAP
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            importance = np.abs(shap_values).mean(axis=0)
            feature_names = X_test.columns
        
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importance / importance.sum(),
            "source": "ml"
        }).sort_values("importance", ascending=False)
        
        return df
    
    except Exception as e:
        logger.error(f"Feature importance extraction failed: {e}")
        return pd.DataFrame()

# In show_ml_prediction():
model = load_ml_model()
if model:
    ml_fi = extract_feature_importance(model)
    st.session_state["ml_feature_importance"] = ml_fi
else:
    ml_fi = generate_dummy_feature_importance("ml")  # Fallback
```

**Action Items:**
- [ ] Ensure model has `feature_names_` attribute
- [ ] Test feature importance extraction
- [ ] Add SHAP-based explanations for non-tree models
- [ ] Visualize with SHAP force plots

---

## Phase 5: Data Pipeline & Logging

### Task 5.1: Add Audit Logging

**New Function:**
```python
def log_prediction(input_features: Dict[str, float], prediction: float, 
                  project_id: Optional[str] = None) -> None:
    """Log prediction to PostgreSQL audit table."""
    try:
        from sqlalchemy import create_engine, text
        from datetime import datetime
        
        engine = create_engine(AppConfig.DB_URL)
        
        insert_query = f"""
            INSERT INTO prediction_logs 
            (project_id, input_features, predicted_logNf, created_at)
            VALUES ('{project_id}', '{json.dumps(input_features)}', {prediction}, '{datetime.now()}')
        """
        
        with engine.connect() as conn:
            conn.execute(text(insert_query))
            conn.commit()
        
        logger.info(f"Prediction logged: logNf={prediction:.3f}")
    
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")
```

**Action Items:**
- [ ] Create PostgreSQL table: `prediction_logs(id, project_id, input_features, predicted_logNf, created_at)`
- [ ] Call `log_prediction()` after every prediction
- [ ] Query logs for model performance analysis

### Task 5.2: Data Versioning

**Add to Session State:**
```python
def init_session_state() -> None:
    defaults = {
        # ... existing ...
        "data_version": None,  # Track which dataset version is loaded
        "model_version": None,  # Track which model is in use
    }
```

**Action Items:**
- [ ] Tag datasets with version/timestamp on import
- [ ] Track model version used for predictions
- [ ] Enable rollback to previous versions

---

## Phase 6: UI Polish

### Task 6.1: Add Material/Processing Route Filters

**In Executive Dashboard:**
```python
cols = st.columns(3)
with cols[0]:
    selected_routes = st.multiselect(
        "Filter by Processing Route",
        data["route_id"].unique(),
        default=data["route_id"].unique()
    )
    filtered_data = data[data["route_id"].isin(selected_routes)]

with cols[1]:
    selected_temps = st.multiselect(
        "Filter by Temperature",
        data["temperature_c"].unique(),
        default=data["temperature_c"].unique()
    )
    filtered_data = filtered_data[filtered_data["temperature_c"].isin(selected_temps)]

with cols[2]:
    st.info(f"Showing {len(filtered_data)} of {len(data)} records")
```

**Action Items:**
- [ ] Add Material Type filter
- [ ] Add Date Range filter (for time-series analysis)
- [ ] Add Temperature filter
- [ ] Add TSA range filter

### Task 6.2: Export Functionality

**New Function:**
```python
def export_analysis_results(data: pd.DataFrame, 
                           analysis_type: str) -> bytes:
    """Export analysis results as CSV/Excel."""
    if analysis_type == "csv":
        return data.to_csv(index=False).encode()
    elif analysis_type == "excel":
        import openpyxl
        buffer = BytesIO()
        data.to_excel(buffer, index=False, sheet_name="Results")
        return buffer.getvalue()

# In each analysis section:
col1, col2 = st.columns(2)
with col1:
    csv = export_analysis_results(results, "csv")
    st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")
with col2:
    excel = export_analysis_results(results, "excel")
    st.download_button("📥 Download Excel", excel, "results.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
```

**Action Items:**
- [ ] Add CSV export for each analysis
- [ ] Add Excel export with formatted sheets
- [ ] Add PDF report generation (use reportlab)

---

## Validation Checklist

### Before Going Live:

- [ ] **Data Integrity:**
  - [ ] Verify all columns in PostgreSQL tables
  - [ ] Check for NULL handling
  - [ ] Test with 100k+ row datasets
  - [ ] Validate data types (int vs float, timestamps)

- [ ] **Model Performance:**
  - [ ] Verify R², RMSE metrics on holdout test set
  - [ ] Check prediction ranges make physical sense
  - [ ] Test edge cases (very small/large grains, extreme temperatures)
  - [ ] Compare model predictions vs. experimental data

- [ ] **Database:**
  - [ ] Connection pooling working (test under load)
  - [ ] Query performance acceptable (<1s for UI operations)
  - [ ] Backup/recovery procedures tested
  - [ ] Audit logging functioning

- [ ] **UI/UX:**
  - [ ] All pages load without errors
  - [ ] Charts render correctly
  - [ ] Error messages are clear and actionable
  - [ ] Status panel updates correctly
  - [ ] Mobile responsiveness (if needed)

- [ ] **Logging & Monitoring:**
  - [ ] Logs written to file/database
  - [ ] Error notifications configured
  - [ ] Performance metrics tracked
  - [ ] User action audit trail complete

---

## Deployment Checklist

- [ ] Secrets management (DB credentials in environment variables)
- [ ] Streamlit config: `streamlit_config.toml` with production settings
- [ ] Docker image built and tested
- [ ] Reverse proxy (nginx) configured for HTTPS
- [ ] Load testing performed
- [ ] Backup strategy verified
- [ ] Monitoring/alerting set up
- [ ] Team trained on dashboard usage

---

## Reference: File Organization

```
fatigue-dbms-dashboard/
├── streamlit_app_enhanced.py          # Main app (enhanced)
├── streamlit_config.toml              # Streamlit config
├── config.py                          # DB credentials (git-ignored)
├── requirements.txt                   # Python dependencies
├── models/
│   ├── rf_fatigue_life.pkl           # Pre-trained RandomForest
│   └── xgb_fatigue_life.pkl          # Pre-trained XGBoost
├── scripts/
│   ├── train_models.py               # ML training pipeline
│   ├── import_csv_data.py            # CSV → PostgreSQL ETL
│   └── compute_statistics.py         # Phase 1–3 analyses
├── tests/
│   ├── test_data_validation.py
│   ├── test_predictions.py
│   └── test_integration.py
└── docs/
    ├── ENHANCEMENTS.md               # Enhancement summary
    └── INTEGRATION.md                # This file
```

---

## Quick Integration Commands

```bash
# Set up environment
export DB_USER="fatigue_user"
export DB_PASS="your_secure_password"
export DB_HOST="localhost"
export DB_PORT="5432"

# Install dependencies
pip install -r requirements.txt

# Run enhanced app locally
streamlit run streamlit_app_enhanced.py

# Run tests
pytest tests/

# Deploy with Docker
docker build -t fatigue-dbms-dashboard .
docker run -e DB_USER=$DB_USER -p 8501:8501 fatigue-dbms-dashboard
```

---

## Support & Questions

For each integration task, refer to:
1. Inline code comments (marked with `REPLACE:`, `TODO:`, `WIRE:`)
2. `ENHANCEMENTS.md` for detailed explanations
3. Your PostgreSQL schema documentation
4. ML model training scripts for feature order/normalization

**Good luck with the integration! 🚀**
