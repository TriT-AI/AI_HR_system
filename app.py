from __future__ import annotations

import streamlit as st
import utils  # Shared utilities

# Import view modules
from views import upload_sync, search_match, dashboard

# Apply theme and init session state (centralized here)
utils.apply_theme()
utils.init_session_state()
# Custom CSS for enhanced UI
st.markdown("""
    <style>
    /* General styling */
    .stApp { background-color: #f8f9fc; }
    .main { padding: 1rem; }
    h1, h2, h3 { color: #1e3a8a; }
    .stButton>button { background-color: #3b82f6; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #2563eb; }
    .card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 1rem; }
    .metric-card { background: #eff6ff; border-radius: 8px; padding: 1rem; text-align: center; }
    .metric-value { font-size: 1.5rem; font-weight: bold; color: #1e40af; }
    .metric-label { font-size: 0.9rem; color: #64748b; }
    .tag { background: #dbeafe; color: #1e40af; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.875rem; margin: 0.25rem; display: inline-block; }
    .warning-tag { background: #fef3c7; color: #92400e; }
    .progress-bar { height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
    .progress-fill { height: 100%; background: linear-gradient(to right, #22c55e, #3b82f6); }
    .footer { text-align: center; color: #64748b; font-size: 0.875rem; margin-top: 2rem; padding: 1rem 0; border-top: 1px solid #e2e8f0; }
    .highlight { background: #dbeafe; padding: 0.5rem; border-radius: 4px; }
    .insight-card { background: #f0fdf4; border-left: 4px solid #22c55e; padding: 1rem; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)
# Sidebar Navigation (handled here in app.py)
with st.sidebar:
    page = st.radio(
        "Navigation",
        ["🏠 Home", "⬆️ Upload & Sync", "🔍 Search & Match", "📊 Dashboard"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Powered by AI • Secure & Local Processing")

# Main Content: Render based on selection
if page == "🏠 Home":
    stats = st.session_state.db.get_database_stats()
    utils.kpi_row(stats)
    st.header("Welcome to HR Resume Processing")
    st.markdown("""
        Streamline your recruitment workflow:
        - **Upload & Sync**: Extract data from CVs automatically
        - **Search & Match**: Find and rank talent
        - **Dashboard**: View analytics and stats
    """)
    st.markdown("<div class='footer'>© 2025 HR System</div>", unsafe_allow_html=True)

elif page == "⬆️ Upload & Sync":
    upload_sync.main()

elif page == "🔍 Search & Match":
    search_match.main()

elif page == "📊 Dashboard":
    dashboard.main()
