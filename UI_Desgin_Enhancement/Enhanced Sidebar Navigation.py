def render_enhanced_sidebar() -> None:
    """Create a professional sidebar with workflow progress"""
    
    with st.sidebar:
        # Company branding
        st.markdown(f"""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #dee2e6;">
            <span style="font-size: 2rem;">⚙️</span>
            <h3 style="margin: 0; color: #0066CC;">{AppConfig.COMPANY_BRANDING}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Workflow Progress")
        
        # Workflow steps with status indicators
        steps = ["Upload", "Validate", "Analyze", "Predict"]
        icons = ["📤", "✅", "📊", "🤖"]
        
        workflow_status = {
            "upload": st.session_state.get("data_uploaded", False),
            "validate": st.session_state.get("validation_confirmed", False),
            "analyze": st.session_state.get("stats_results") is not None,
            "predict": st.session_state.get("ml_results") is not None
        }
        
        # Create step indicator HTML
        step_html = '<div class="step-indicator">'
        for i, (step, icon) in enumerate(zip(steps, icons)):
            status = "completed" if list(workflow_status.values())[i] else "pending"
            step_html += f"""
            <div class="step {status}">
                <div class="step-circle">{icon}</div>
                <div>{step}</div>
            </div>
            """
        step_html += '</div>'
        
        st.markdown(step_html, unsafe_allow_html=True)
        st.markdown("---")
        
        # Navigation with icons
        st.markdown("### Navigation")
        page_icons = {
            "Executive Dashboard": "📊",
            "Data Lineage": "📁", 
            "Statistical Modelling": "📈",
            "Machine Learning": "🤖"
        }
        
        selected_page = st.radio(
            "Go to",
            list(page_icons.keys()),
            format_func=lambda x: f"{page_icons[x]} {x}",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats summary
        if st.session_state.get("current_data") is not None:
            df = st.session_state["current_data"]
            st.markdown("### Dataset Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Samples", len(df))
            with col2:
                if "route_family" in df.columns:
                    st.metric("Routes", df["route_family"].nunique())
        
        # Help section in sidebar
        with st.expander("📚 Help & Resources", expanded=False):
            st.markdown("""
            **Quick Guide:**
            1. Upload your fatigue test CSV
            2. Validate schema requirements
            3. Explore statistical patterns
            4. Generate ML predictions
            
            **Required Columns:**
            - specimen_id, route_family
            - YS_MPa, grain_size_um
            - Nf, TSA, frequency_Hz
            - PSA_mean, mean_stress_mean
            
            **Support:** support@fatigue-lab.com
            """)
        
        # Version info
        st.markdown("---")
        st.markdown(f"""
        <div style="text-align: center; color: #6c757d; font-size: 0.8rem;">
            {AppConfig.APP_TITLE} {AppConfig.APP_VERSION}<br>
            © 2024 {AppConfig.COMPANY_BRANDING}
        </div>
        """, unsafe_allow_html=True)
        
        return selected_page