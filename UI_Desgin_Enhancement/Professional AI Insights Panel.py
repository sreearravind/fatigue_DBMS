def render_enhanced_ai_summary(df: pd.DataFrame) -> None:
    """Create a polished AI insights panel with metallurgical context"""
    
    st.markdown("### 🧠 AI-Powered Insights")
    
    # Create tabs for different summary views
    summary_tab1, summary_tab2 = st.tabs(["📋 Executive Summary", "💬 Ask AI"])
    
    with summary_tab1:
        # Generate insights based on data
        insights = []
        
        # Dataset overview
        insights.append(f"**Dataset Overview:** {len(df)} specimens across {df['route_family'].nunique() if 'route_family' in df.columns else 'multiple'} processing routes")
        
        # Fatigue life statistics
        if "Nf" in df.columns:
            mean_nf = df["Nf"].mean()
            median_nf = df["Nf"].median()
            insights.append(f"**Fatigue Life:** Mean = {mean_nf:,.0f} cycles, Median = {median_nf:,.0f} cycles")
        
        # Key correlations
        if "Nf" in df.columns and "grain_size_um" in df.columns:
            grain_corr = df["Nf"].corr(df["grain_size_um"])
            if pd.notna(grain_corr):
                direction = "increases" if grain_corr > 0 else "decreases"
                insights.append(f"**Grain Size Effect:** {'+' if grain_corr > 0 else ''}{grain_corr:.3f} correlation - fatigue life {direction} with grain size")
        
        if "Nf" in df.columns and "YS_MPa" in df.columns:
            ys_corr = df["Nf"].corr(df["YS_MPa"])
            if pd.notna(ys_corr):
                direction = "increases" if ys_corr > 0 else "decreases"
                insights.append(f"**Yield Strength Effect:** {'+' if ys_corr > 0 else ''}{ys_corr:.3f} correlation - fatigue life {direction} with yield strength")
        
        # Route comparisons
        if "route_family" in df.columns and "Nf" in df.columns:
            route_stats = df.groupby("route_family")["Nf"].agg(["mean", "std"]).round(0)
            best_route = route_stats["mean"].idxmax()
            best_mean = route_stats.loc[best_route, "mean"]
            insights.append(f"**Best Performing Route:** {best_route} (Mean Nf = {best_mean:,.0f} cycles)")
        
        # Recommendations
        insights.append("**Recommendations:**")
        if "grain_size_um" in df.columns and "Nf" in df.columns:
            if grain_corr < -0.3:
                insights.append("- ⚙️ Refine grain structure through thermomechanical processing to improve fatigue life")
        if "YS_MPa" in df.columns and "Nf" in df.columns:
            if ys_corr > 0.3:
                insights.append("- 🔧 Optimize precipitation hardening to increase yield strength and fatigue resistance")
        
        # Display insights in a nice card layout
        st.markdown("""
        <div class="info-card" style="background: linear-gradient(135deg, #f8f9fa, #ffffff);">
        """, unsafe_allow_html=True)
        
        for insight in insights:
            st.markdown(f"• {insight}")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add visualization of key insights
        if "route_family" in df.columns and "Nf" in df.columns:
            st.markdown("#### Route Performance Comparison")
            route_perf = df.groupby("route_family")["Nf"].agg(["mean", "sem"]).reset_index()
            route_perf.columns = ["Route", "Mean Nf", "Error"]
            
            route_chart = alt.Chart(route_perf).mark_bar(color="#0066CC").encode(
                x=alt.X("Route:N", sort="-y"),
                y=alt.Y("Mean Nf:Q", title="Mean Cycles to Failure")
            ).properties(height=300)
            
            error_bars = alt.Chart(route_perf).mark_errorbar(extent="ci").encode(
                x="Route:N",
                y=alt.Y("Mean Nf:Q", title="Mean Nf"),
                yError="Error:Q"
            )
            
            st.altair_chart(route_chart + error_bars, use_container_width=True)
    
    with summary_tab2:
        st.markdown("#### Ask the Materials Intelligence Assistant")
        
        user_query = st.text_input(
            "Ask a question about your fatigue data:",
            placeholder="e.g., What factors most influence fatigue life? How does grain size affect Nf?"
        )
        
        if user_query:
            with st.spinner("Analyzing your question..."):
                # Generate contextual response based on query keywords
                response_parts = ["Based on your fatigue data analysis:"]
                
                if "grain" in user_query.lower() and "grain_size_um" in df.columns:
                    if "Nf" in df.columns:
                        corr = df["Nf"].corr(df["grain_size_um"])
                        response_parts.append(f"📊 Grain size shows a {corr:+.3f} correlation with fatigue life. " + 
                                             ("Smaller grains generally improve fatigue resistance due to Hall-Petch strengthening." 
                                              if corr < 0 else "Larger grains may be beneficial in this dataset."))
                
                if "route" in user_query.lower() and "route_family" in df.columns:
                    route_means = df.groupby("route_family")["Nf"].mean().sort_values(ascending=False)
                    top_route = route_means.index[0]
                    response_parts.append(f"🏭 The {top_route} processing route shows the highest average fatigue life " +
                                         f"({route_means.iloc[0]:,.0f} cycles).")
                
                if "stress" in user_query.lower() or "strain" in user_query.lower():
                    if "mean_stress_mean" in df.columns and "Nf" in df.columns:
                        stress_corr = df["Nf"].corr(df["mean_stress_mean"])
                        response_parts.append(f"⚡ Mean stress has a {stress_corr:+.3f} correlation with fatigue life. " +
                                             "Higher mean stresses typically reduce fatigue life due to increased damage per cycle.")
                
                if len(response_parts) == 1:
                    response_parts.append("Based on available data, focus on grain size optimization and processing route selection for improved fatigue performance.")
                
                # Display response in a chat bubble
                st.markdown(f"""
                <div style="background: #e9ecef; padding: 1.5rem; border-radius: 15px; margin-top: 1rem;">
                    <strong>🤖 Assistant</strong><br>
                    {chr(10).join(response_parts)}
                </div>
                """, unsafe_allow_html=True)
                
                # Store in session state
                set_last_operation("AI query processed")