def render_validation_dashboard(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a comprehensive validation dashboard with visual feedback"""
    
    st.markdown("### 🔍 Schema Validation Dashboard")
    
    report = build_validation_report(df)
    st.session_state["validation_report"] = report
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        required_present = len(REQUIRED_COLUMNS) - len(report["missing_required"])
        st.metric(
            "Required Columns",
            f"{required_present}/{len(REQUIRED_COLUMNS)}",
            delta=None,
            help="Columns essential for analysis"
        )
    
    with col2:
        optional_present = len(OPTIONAL_COLUMNS) - len(report["missing_optional"])
        st.metric(
            "Optional Columns",
            f"{optional_present}/{len(OPTIONAL_COLUMNS)}",
            delta=None,
            help="Additional recommended columns"
        )
    
    with col3:
        missing_cells = sum(report["missing_values"].values())
        completeness = ((len(df) * len(df.columns) - missing_cells) / (len(df) * len(df.columns)) * 100)
        st.metric(
            "Data Completeness",
            f"{completeness:.1f}%",
            delta=None,
            help="Percentage of non-null values"
        )
    
    with col4:
        validation_status = "✅ Pass" if report["required_ok"] else "❌ Fail"
        st.metric(
            "Validation Status",
            validation_status,
            delta=None,
            help="All required columns must be present"
        )
    
    # Detailed column validation
    st.markdown("#### Column Validation Details")
    
    validation_data = []
    for col in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        status = "✅" if col in df.columns else "❌" if col in REQUIRED_COLUMNS else "⚠️"
        data_type = str(df[col].dtype) if col in df.columns else "Missing"
        missing = df[col].isna().sum() if col in df.columns else len(df)
        
        validation_data.append({
            "Status": status,
            "Column": col,
            "Required": "Yes" if col in REQUIRED_COLUMNS else "No",
            "Type": data_type,
            "Missing": missing,
            "Present %": f"{(1 - missing/len(df))*100:.1f}%" if col in df.columns else "0%"
        })
    
    validation_df = pd.DataFrame(validation_data)
    
    # Color-coded dataframe
    def color_status(val):
        if val == "✅":
            return "background-color: #d4edda"
        elif val == "❌":
            return "background-color: #f8d7da"
        elif val == "⚠️":
            return "background-color: #fff3cd"
        return ""
    
    styled_df = validation_df.style.applymap(color_status, subset=["Status"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Data type issues
    if report["dtype_issues"]:
        st.warning("⚠️ **Data Type Mismatches Found**")
        for col, dtype in report["dtype_issues"].items():
            expected = "numeric" if col not in ["specimen_id", "route_family"] else "text/category"
            st.markdown(f"- `{col}`: expected **{expected}**, found **{dtype}**")
    
    return report