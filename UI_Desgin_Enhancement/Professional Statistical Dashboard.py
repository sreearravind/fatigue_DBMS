def render_enhanced_statistical_modelling(df: pd.DataFrame) -> None:
    """Create a polished statistical analysis interface"""
    
    st.markdown("### 📊 Statistical Analysis")
    
    # Create tabs for different analysis views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Summary Statistics",
        "📈 Distributions", 
        "🔗 Correlations",
        "🎯 Influence Analysis"
    ])
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    with tab1:
        # Descriptive statistics with formatting
        desc = df[numeric_cols].describe().T
        
        # Add additional metrics
        desc["skew"] = df[numeric_cols].skew()
        desc["kurtosis"] = df[numeric_cols].kurtosis()
        desc["missing_%"] = (df[numeric_cols].isna().sum() / len(df) * 100).round(2)
        
        # Format for display
        desc_display = desc.round(2)
        
        st.markdown("#### Descriptive Statistics")
        st.dataframe(
            desc_display,
            use_container_width=True,
            column_config={
                "count": st.column_config.NumberColumn("Count", format="%d"),
                "mean": st.column_config.NumberColumn("Mean", format="%.2f"),
                "std": st.column_config.NumberColumn("Std Dev", format="%.2f"),
                "min": st.column_config.NumberColumn("Min", format="%.2f"),
                "25%": st.column_config.NumberColumn("25%", format="%.2f"),
                "50%": st.column_config.NumberColumn("50%", format="%.2f"),
                "75%": st.column_config.NumberColumn("75%", format="%.2f"),
                "max": st.column_config.NumberColumn("Max", format="%.2f"),
            }
        )
    
    with tab2:
        st.markdown("#### Feature Distributions")
        
        # Select feature for distribution
        selected_feature = st.selectbox(
            "Select feature to visualize",
            options=numeric_cols,
            index=numeric_cols.index("Nf") if "Nf" in numeric_cols else 0
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram with KDE
            hist_data = df[selected_feature].dropna()
            
            # Create histogram with Altair
            hist_chart = alt.Chart(df).mark_bar(
                opacity=0.7,
                color="#0066CC"
            ).encode(
                x=alt.X(f"{selected_feature}:Q", bin=alt.Bin(maxbins=30)),
                y=alt.Y("count()", title="Frequency"),
                tooltip=["count()"]
            ).properties(
                height=300,
                title=f"Distribution of {selected_feature}"
            )
            
            # Add KDE if enough data points
            if len(hist_data) > 10:
                kde = alt.Chart(df).transform_density(
                    selected_feature,
                    as_=[selected_feature, "density"],
                ).mark_line(color="red").encode(
                    x=f"{selected_feature}:Q",
                    y="density:Q"
                )
                st.altair_chart(hist_chart + kde, use_container_width=True)
            else:
                st.altair_chart(hist_chart, use_container_width=True)
        
        with col2:
            # Box plot for the same feature
            if "route_family" in df.columns:
                box_chart = alt.Chart(df).mark_boxplot().encode(
                    x=alt.X("route_family:N", title="Route Family"),
                    y=alt.Y(f"{selected_feature}:Q", title=selected_feature),
                    color=alt.Color("route_family:N", legend=None)
                ).properties(
                    height=300,
                    title=f"{selected_feature} by Route Family"
                )
                st.altair_chart(box_chart, use_container_width=True)
    
    with tab3:
        st.markdown("#### Correlation Analysis")
        
        # Feature selection for correlation
        selected_features = st.multiselect(
            "Select features for correlation matrix",
            options=numeric_cols,
            default=numeric_cols[:min(8, len(numeric_cols))]
        )
        
        if len(selected_features) >= 2:
            corr = df[selected_features].corr()
            
            # Create enhanced correlation heatmap
            corr_long = corr.reset_index().melt(id_vars="index", var_name="feature", value_name="correlation")
            
            # Add correlation values as text
            base = alt.Chart(corr_long).encode(
                x="index:N",
                y="feature:N"
            )
            
            heatmap = base.mark_rect().encode(
                color=alt.Color("correlation:Q", 
                              scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                              title="Correlation")
            )
            
            text = base.mark_text(baseline="middle").encode(
                text=alt.Text("correlation:Q", format=".2f"),
                color=alt.condition(
                    alt.datum.correlation > 0.5,
                    alt.value("white"),
                    alt.value("black")
                )
            )
            
            st.altair_chart(heatmap + text, use_container_width=True)
            
            # Show correlation with target
            if "Nf" in selected_features:
                st.markdown("#### Top Correlations with Nf")
                target_corr = corr["Nf"].drop("Nf").sort_values(ascending=False)
                
                corr_df = pd.DataFrame({
                    "Feature": target_corr.index,
                    "Correlation": target_corr.values,
                    "|Correlation|": abs(target_corr.values)
                }).sort_values("|Correlation|", ascending=False).head(10)
                
                # Bar chart of top correlations
                corr_chart = alt.Chart(corr_df).mark_bar().encode(
                    x=alt.X("|Correlation|:Q", title="Absolute Correlation"),
                    y=alt.Y("Feature:N", sort="-x"),
                    color=alt.condition(
                        alt.datum.Correlation > 0,
                        alt.value("#28A745"),
                        alt.value("#DC3545")
                    ),
                    tooltip=["Feature", "Correlation"]
                ).properties(height=300)
                
                st.altair_chart(corr_chart, use_container_width=True)
    
    with tab4:
        st.markdown("#### Feature Influence Ranking")
        
        if "Nf" in df.columns:
            # Prepare features for influence analysis
            feature_cols = [c for c in numeric_cols if c != "Nf" and c != "log_Nf"]
            
            # Calculate correlations with Nf
            influence_data = []
            for feature in feature_cols:
                if feature in df.columns:
                    corr_val = df[feature].corr(df["Nf"])
                    if pd.notna(corr_val):
                        influence_data.append({
                            "feature": feature,
                            "correlation": corr_val,
                            "abs_correlation": abs(corr_val),
                            "influence": "Positive" if corr_val > 0 else "Negative"
                        })
            
            influence_df = pd.DataFrame(influence_data).sort_values("abs_correlation", ascending=False)
            
            # Create influence chart
            influence_chart = alt.Chart(influence_df.head(15)).mark_bar().encode(
                x=alt.X("abs_correlation:Q", title="Influence Magnitude"),
                y=alt.Y("feature:N", sort="-x", title="Feature"),
                color=alt.Color("influence:N", 
                              scale=alt.Scale(domain=["Positive", "Negative"],
                                            range=["#28A745", "#DC3545"])),
                tooltip=["feature", "correlation"]
            ).properties(
                height=400,
                title="Top 15 Features Influencing Fatigue Life (Nf)"
            )
            
            st.altair_chart(influence_chart, use_container_width=True)
            
            # Store results
            st.session_state["stats_results"] = {
                "descriptive": desc.to_dict(),
                "target_mean": float(df["Nf"].mean()),
                "target_cov": float((df["Nf"].std() / df["Nf"].mean() * 100)),
                "influence_ranking": influence_df.to_dict("records")
            }