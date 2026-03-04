def apply_custom_styling() -> None:
    """Apply professional metallurgy-themed custom CSS"""
    st.markdown("""
    <style>
        /* Global Typography */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Headers */
        h1 {
            color: #0066CC;
            font-weight: 700;
            letter-spacing: -0.02em;
            border-bottom: 3px solid #0066CC;
            padding-bottom: 0.5rem;
            margin-bottom: 2rem;
        }
        
        h2 {
            color: #343A40;
            font-weight: 600;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        h3 {
            color: #495057;
            font-weight: 600;
            font-size: 1.3rem;
        }
        
        /* Metric Cards */
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem;
            font-weight: 700;
            color: #0066CC;
        }
        
        div[data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #6C757D;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-size: 0.85rem;
        }
        
        /* Status Indicators */
        .status-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-right: 0.5rem;
        }
        
        .status-badge.connected {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .status-badge.disconnected {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        /* Info Cards */
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid #e9ecef;
            margin-bottom: 1rem;
        }
        
        /* Progress Steps */
        .step-indicator {
            display: flex;
            justify-content: space-between;
            margin: 2rem 0;
            padding: 0;
        }
        
        .step {
            flex: 1;
            text-align: center;
            position: relative;
        }
        
        .step.completed .step-circle {
            background: #28A745;
            color: white;
        }
        
        .step.active .step-circle {
            background: #0066CC;
            color: white;
            box-shadow: 0 0 0 3px rgba(0,102,204,0.2);
        }
        
        .step-circle {
            width: 40px;
            height: 40px;
            background: #e9ecef;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 0.5rem;
            font-weight: 700;
            color: #6c757d;
        }
        
        /* Tooltips */
        .tooltip-icon {
            color: #17a2b8;
            cursor: help;
            border-bottom: 1px dashed #17a2b8;
        }
        
        /* Buttons */
        .stButton > button {
            font-weight: 600;
            border-radius: 6px;
            transition: all 0.2s;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Sidebar enhancements */
        section[data-testid="stSidebar"] {
            background-color: #f8f9fa;
            border-right: 1px solid #dee2e6;
        }
        
        section[data-testid="stSidebar"] .css-1d391kg {
            padding-top: 2rem;
        }
        
        /* Dataframes */
        .dataframe-container {
            border: 1px solid #dee2e6;
            border-radius: 8px;
            overflow: hidden;
        }
        
        /* Alerts and notifications */
        .stAlert {
            border-radius: 8px;
            border-left-width: 4px;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 1rem;
            color: #6c757d;
            font-size: 0.85rem;
            border-top: 1px solid #dee2e6;
            margin-top: 3rem;
        }
    </style>
    """, unsafe_allow_html=True)