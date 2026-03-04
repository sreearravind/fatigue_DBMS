def main() -> None:
    """Enhanced main application with professional UI"""
    
    # Page configuration
    st.set_page_config(
        page_title=AppConfig.APP_TITLE,
        page_icon=AppConfig.APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom styling
    apply_custom_styling()
    
    # Initialize session state
    init_session_state()
    
    # Connect to database
    connect_to_database()
    
    # Render professional header (replaces old top bar)
    render_professional_header()
    
    # Get selected page from enhanced sidebar
    selected_page = render_enhanced_sidebar()
    
    # Main content area
    if selected_page == "Executive Dashboard":
        st.markdown("## 📊 Executive Dashboard")
        
        # Data acquisition section
        df = render_enhanced_upload_section()
        
        if df is None:
            df = get_active_dataset()
        
        # Validation section
        if df is not None:
            report = render_validation_dashboard(df)
            
            # Confirm button with proper styling
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("✅ Confirm & Proceed to Analysis", use_container_width=True):
                    if report["required_ok"]:
                        st.session_state["validation_confirmed"] = True
                        set_last_operation("Validation confirmed")
                        st.success("Validation successful! Proceed to Statistical Modelling.")
                    else:
                        st.error("Cannot proceed: missing required columns")
            
            # Quick KPIs
            render_enhanced_route_kpis(df)
            
            # AI Summary
            render_enhanced_ai_summary(df)
    
    elif selected_page == "Data Lineage":
        st.markdown("## 📁 Data Lineage & Validation")
        df = get_active_dataset()
        render_validation_dashboard(df)
    
    elif selected_page == "Statistical Modelling":
        st.markdown("## 📈 Statistical Modelling")
        if not st.session_state.get("validation_confirmed", False):
            st.warning("⚠️ Please complete Data Validation first")
            if st.button("Go to Data Lineage"):
                st.session_state["navigation"] = "Data Lineage"
        else:
            df = get_active_dataset()
            render_enhanced_statistical_modelling(df)
    
    elif selected_page == "Machine Learning":
        st.markdown("## 🤖 Machine Learning")
        if not st.session_state.get("validation_confirmed", False):
            st.warning("⚠️ Please complete Data Validation first")
            if st.button("Go to Data Lineage"):
                st.session_state["navigation"] = "Data Lineage"
        else:
            df = get_active_dataset()
            render_enhanced_machine_learning(df)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div class="footer">
        <p>🔬 {AppConfig.APP_TITLE} {AppConfig.APP_VERSION} | Advanced Fatigue Analysis for Metallurgical Applications</p>
        <p style="font-size: 0.7rem;">Last operation: {st.session_state.get('last_operation', 'App started')}</p>
    </div>
    """, unsafe_allow_html=True)