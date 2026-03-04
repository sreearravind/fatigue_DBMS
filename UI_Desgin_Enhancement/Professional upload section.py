def render_enhanced_upload_section() -> Optional[pd.DataFrame]:
    """Create a polished data upload interface with template support"""
    
    st.markdown("### 📂 Data Acquisition")
    
    # Template download button
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("📥 Download Template", use_container_width=True):
            # Create template DataFrame
            template_df = pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS[:5])
            template_df.loc[0] = ["S001", "T6A", 250, 35, 10000, 0.4, 0.5, 0.35, 120] + [np.nan]*5
            csv = template_df.to_csv(index=False)
            st.download_button(
                label="Save Template CSV",
                data=csv,
                file_name="fatigue_data_template.csv",
                mime="text/csv"
            )
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>Expected Data Format</h4>
            <p>Upload CSV files containing fatigue test results with the following columns:</p>
        </div>
        """, unsafe_allow_html=True)
    
    # File uploader with enhanced UI
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload fatigue test data in CSV format. Maximum file size: 200MB",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("📊 Processing uploaded file..."):
                uploaded_df = pd.read_csv(uploaded_file)
                uploaded_df = normalize_columns(uploaded_df)
                report = build_validation_report(uploaded_df)
                
                st.session_state["current_data"] = uploaded_df
                st.session_state["data_uploaded"] = True
                st.session_state["validation_report"] = report
                
                # Success message with file details
                st.markdown(f"""
                <div class="info-card" style="border-left: 4px solid #28A745;">
                    <strong>✅ File loaded successfully</strong><br>
                    Filename: {uploaded_file.name}<br>
                    Rows: {len(uploaded_df):,} | Columns: {len(uploaded_df.columns)}<br>
                    File size: {uploaded_file.size / 1024:.1f} KB
                </div>
                """, unsafe_allow_html=True)
                
                return uploaded_df
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return None
    
    # Show sample data if no file uploaded
    else:
        st.markdown("""
        <div class="info-card" style="background: #e9ecef;">
            <strong>📋 Sample Data Preview</strong><br>
            <small>Showing demo dataset. Upload your own CSV to begin analysis.</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Show schema preview
        schema_df = pd.DataFrame({
            "Required": REQUIRED_COLUMNS[:5] + [""]*(len(REQUIRED_COLUMNS)-5),
            "Type": ["Text/ID", "Category", "Numeric"]*3,
            "Example": ["S001", "T6A", "250 MPa"]
        })
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
        
        return None