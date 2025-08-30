from __future__ import annotations

import asyncio
import base64
import logging
import nest_asyncio
import threading
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any
import json
import pandas as pd
import altair as alt
from PIL import Image
import streamlit as st

from cv_processor import CVProcessor
from database import ResumeDatabase
from job_matcher import JobMatcher
from google_drive_processor import GoogleDriveProcessor
from config import get_settings

# App Config + Theming (shared across pages)
def apply_theme():
    st.set_page_config(
        page_title="HR Resume Processing System",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
        <style>
        /* Your custom CSS here - copy the entire <style> block from original app.py */
        </style>
    """, unsafe_allow_html=True)

# Session Singletons (initialize once)
def init_session_state():
    if "db" not in st.session_state:
        st.session_state.db = ResumeDatabase("hr_resume_system.db")
    if "processor" not in st.session_state:
        st.session_state.processor = CVProcessor()
    if "job_matcher" not in st.session_state:
        st.session_state.job_matcher = JobMatcher(st.session_state.db)
    if "drive_processor" not in st.session_state:
        st.session_state.drive_processor = GoogleDriveProcessor(
            st.session_state.processor, st.session_state.db
        )

# Helpers (copy all helper functions from original app.py here)
# e.g., save_upload_and_cache_bytes, embed_pdf_from_bytes, run_async, kpi_row, etc.
# ... (paste the entire Helpers section)
# =============================================================================
# Helpers
# =============================================================================
def save_upload_and_cache_bytes(uploaded) -> tuple[Path, bytes]:
    tmp_dir = Path("temp_uploads")
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / uploaded.name
    content = uploaded.read()
    with open(tmp_path, "wb") as f:
        f.write(content)
    return tmp_path, content

def embed_pdf_from_bytes(bytes_data, height=600):
    b64 = base64.b64encode(bytes_data).decode('utf-8')
    html = f"<iframe src='data:application/pdf;base64,{b64}' width='100%' height='{height}px'></iframe>"
    st.markdown(html, unsafe_allow_html=True)

def run_async(coroutine):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        result_holder = {}
        def thread_target():
            try:
                result_holder['result'] = asyncio.run(coroutine)
            except Exception as exc:
                result_holder['error'] = exc
        th = threading.Thread(target=thread_target)
        th.start()
        th.join()
        if 'error' in result_holder:
            raise result_holder['error']
        return result_holder.get('result')
    else:
        return asyncio.run(coroutine)

def kpi_row(stats: Dict[str, int]):
    cols = st.columns(5)
    icons = ["👥", "💼", "🎓", "🛠️", "📋"]
    keys = ["total_candidates", "total_work_experiences", "total_educations", "total_skills", "total_projects"]
    labels = ["Candidates", "Work Exps", "Educations", "Skills", "Projects"]
    for col, icon, key, label in zip(cols, icons, keys, labels):
        with col:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats.get(key, 0)}</div>
                    <div class="metric-label">{icon} {label}</div>
                </div>
            """, unsafe_allow_html=True)

def display_progress(idx: int, total: int, label: str):
    st.markdown(f"<div class='muted small'>{label} ({idx}/{total})</div>", unsafe_allow_html=True)
    progress = idx / total
    st.markdown(f"""
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress * 100}%"></div>
        </div>
    """, unsafe_allow_html=True)

def export_json(data: Any, label: str, filename: str):
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    st.download_button(
        label=label,
        data=json_str,
        file_name=filename,
        mime="application/json",
    )

def export_csv(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False)
    st.download_button(
        label="⬇️ Export CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )

nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()
