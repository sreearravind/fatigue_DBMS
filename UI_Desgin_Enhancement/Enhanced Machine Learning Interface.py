def render_enhanced_machine_learning(df: pd.DataFrame) -> None:
    """Create a professional ML prediction interface"""
    
    st.markdown("### 🤖 Machine Learning Predictor")
    
    # Model configuration
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select Model",
            options=["Linear Regression (Baseline)", "Random Forest", "XGBoost", "Neural Network"],
            help="Choose the ML algorithm for fatigue life prediction"
        )
    
    with col2:
        test_size = st.slider(
            "Test Split %",
            min_value=10,
            max_value=40,
            value=20,
            help="Percentage of data used for testing"
        )
    
    # Feature engineering section
    st.markdown("#### Feature Configuration")
    
    # Available features
    feature_options = {
        "grain_size_um": "Grain Size (Hall-Petch)",
        "YS_MPa": "Yield Strength",
        "mean_stress_mean": "Mean Stress",
        "PSA_mean": "PSA Mean",
        "frequency_Hz": "Test Frequency",
        "TSA": "Total Strain Amplitude",
        "temperature_C": "Temperature",
        "Hardness_hv": "Hardness",
        "elongation_percent": "Elongation"
    }
    
    # Filter to available columns
    available_features = {k: v for k, v in feature_options.items() if k in df.columns}
    
    selected_features = st.multiselect(
        "Select predictor features",
        options=list(available_features.keys()),
        default=[f for f in ["grain_size_um", "YS_MPa", "mean_stress_mean", "PSA_mean", "TSA"] if f in df.columns],
        format_func=lambda x: f"{x} - {available_features[x]}",
        help="Choose features that influence fatigue life"
    )
    
    # Prepare data
    if len(selected_features) >= 2 and "Nf" in df.columns:
        
        # Handle categorical features
        if "route_family" in df.columns:
            df_encoded = pd.get_dummies(df, columns=["route_family"], prefix="route")
            feature_cols = selected_features + [c for c in df_encoded.columns if c.startswith("route_")]
        else:
            df_encoded = df.copy()
            feature_cols = selected_features
        
        # Remove rows with missing values
        model_df = df_encoded[feature_cols + ["Nf"]].dropna()
        
        if len(model_df) >= 10:
            X = model_df[feature_cols].values
            y = model_df["Nf"].values
            
            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size/100, random_state=42
            )
            
            # Model training based on selection
            with st.spinner("🔄 Training model..."):
                if model_type == "Linear Regression (Baseline)":
                    from sklearn.linear_model import LinearRegression
                    model = LinearRegression()
                elif model_type == "Random Forest":
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                elif model_type == "XGBoost":
                    from xgboost import XGBRegressor
                    model = XGBRegressor(random_state=42)
                else:  # Neural Network
                    from sklearn.neural_network import MLPRegressor
                    model = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=42)
                
                model.fit(X_train, y_train)
                
                # Predictions and metrics
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                from sklearn.metrics import r2_score, mean_squared_error
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train R²", f"{train_r2:.3f}", help="Model fit on training data")
            with col2:
                st.metric("Test R²", f"{test_r2:.3f}", 
                         delta=f"{test_r2 - train_r2:.3f}",
                         delta_color="inverse",
                         help="Model generalization")
            with col3:
                st.metric("Test RMSE", f"{test_rmse:.0f}", help="Prediction error")
            
            # Feature importance
            if hasattr(model, "coef_"):
                importance = np.abs(model.coef_[:len(selected_features)])
                importance_features = selected_features
            elif hasattr(model, "feature_importances_"):
                importance = model.feature_importances_[:len(selected_features)]
                importance_features = selected_features
            else:
                importance = None
            
            if importance is not None:
                st.markdown("#### Feature Importance")
                imp_df = pd.DataFrame({
                    "Feature": importance_features,
                    "Importance": importance
                }).sort_values("Importance", ascending=False)
                
                imp_chart = alt.Chart(imp_df).mark_bar(color="#0066CC").encode(
                    x=alt.X("Importance:Q"),
                    y=alt.Y("Feature:N", sort="-x")
                ).properties(height=300)
                
                st.altair_chart(imp_chart, use_container_width=True)
            
            # Prediction interface
            st.markdown("#### Make Predictions")
            st.markdown("Enter specimen parameters:")
            
            col1, col2 = st.columns(2)
            input_values = {}
            
            with col1:
                for feat in selected_features[:len(selected_features)//2 + 1]:
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    mean_val = float(df[feat].mean())
                    input_values[feat] = st.slider(
                        f"{feat}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        format="%.2f",
                        help=f"Range: {min_val:.2f} - {max_val:.2f}"
                    )
            
            with col2:
                for feat in selected_features[len(selected_features)//2 + 1:]:
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    mean_val = float(df[feat].mean())
                    input_values[feat] = st.slider(
                        f"{feat}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        format="%.2f",
                        help=f"Range: {min_val:.2f} - {max_val:.2f}"
                    )
            
            # Route family selection
            if "route_family" in df.columns:
                route_options = df["route_family"].unique().tolist()
                selected_route = st.selectbox("Route Family", options=route_options)
                
                # One-hot encode route
                for route in route_options:
                    route_col = f"route_{route}"
                    input_values[route_col] = 1 if route == selected_route else 0
            
            # Create prediction array
            X_pred = np.array([[input_values.get(col, 0) for col in feature_cols]])
            
            # Make prediction
            prediction = model.predict(X_pred)[0]
            
            # Display prediction with confidence interval
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #0066CC11, #0066CC22); border-radius: 10px;">
                    <h4 style="color: #6c757d;">Predicted Fatigue Life</h4>
                    <h1 style="color: #0066CC; font-size: 3.5rem;">{prediction:,.0f}</h1>
                    <p>cycles to failure (Nf)</p>
                    <p style="color: #6c757d; font-size: 0.9rem;">95% CI: [{prediction - 2*test_rmse:,.0f}, {prediction + 2*test_rmse:,.0f}]</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Store results
            st.session_state["ml_results"] = {
                "model_type": model_type,
                "prediction": prediction,
                "r2": test_r2,
                "rmse": test_rmse,
                "feature_importance": imp_df.to_dict("records") if importance is not None else None
            }