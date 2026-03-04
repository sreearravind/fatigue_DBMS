def render_professional_header() -> None:
    """Render a polished header with branding and system controls"""
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="font-size: 2.5rem;">🔬</span>
            <div>
                <h1 style="margin: 0; border-bottom: none;">{AppConfig.APP_TITLE}</h1>
                <p style="margin: 0; color: #6c757d; font-size: 0.9rem;">
                    {AppConfig.APP_LOGO_TEXT} | {AppConfig.APP_VERSION}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Project ID display
        project_id = st.session_state.get("current_project_id", "No Active Project")
        st.markdown(f"""
        <div style="background: #e9ecef; padding: 0.5rem 1rem; border-radius: 8px;">
            <small style="color: #6c757d;">PROJECT</small><br>
            <strong style="color: #0066CC;">{project_id}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # System status badges
        connection_status = "Connected" if st.session_state.get("db_connected") else "Not Connected"
        status_class = "connected" if st.session_state.get("db_connected") else "disconnected"
        st.markdown(f"""
        <div style="text-align: right;">
            <span class="status-badge {status_class}">🌐 {connection_status}</span><br>
            <small>License: <strong>{st.session_state.get('license_tier')}</strong></small>
        </div>
        """, unsafe_allow_html=True)