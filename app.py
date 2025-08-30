"""
app.py – Polished Streamlit front-end for HR Resume Processing System
- Modern, user-friendly design with intuitive navigation
- Enhanced layouts with tabs, cards, and progress indicators
- Improved error handling and feedback
"""

from __future__ import annotations

import asyncio
import base64
import logging
import nest_asyncio
import threading
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import altair as alt
from PIL import Image
import streamlit as st

from cv_processor import CVProcessor
from database import ResumeDatabase
from job_matcher import JobMatcher
from google_drive_processor import GoogleDriveProcessor
from config import get_settings

# =============================================================================
# App Config + Theming
# =============================================================================
st.set_page_config(
    page_title="HR Resume Processing System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

nest_asyncio.apply()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
settings = get_settings()

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

# =============================================================================
# Session Singletons
# =============================================================================
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

# =============================================================================
# Sidebar Navigation
# =============================================================================
with st.sidebar:
    page = st.radio(
        "Navigation",
        ["🏠 Home", "⬆️ Upload & Sync", "🔍 Search & Match", "📊 Dashboard"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Powered by AI • Secure & Local Processing")

# =============================================================================
# Main Content
# =============================================================================
stats = st.session_state.db.get_database_stats()
kpi_row(stats)

if page == "🏠 Home":
    st.header("Welcome to HR Resume Processing")
    st.markdown("""
        Streamline your recruitment workflow:
        - **Upload & Sync**: Extract data from CVs automatically
        - **Search & Match**: Find and rank talent
        - **Dashboard**: View analytics and stats
    """)

elif page == "⬆️ Upload & Sync":
    tab1, tab2 = st.tabs(["📤 Upload CVs", "☁️ Drive Sync"])

    with tab1:
        st.header("Upload CVs")
        uploaded_files = st.file_uploader(
            "Drop files here",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Supports batch processing of resumes",
        )

        if uploaded_files:
            st.success(f"{len(uploaded_files)} files uploaded. Ready to process?")
            if st.button("Process Now", type="primary"):
                results = []
                with st.spinner("Processing files..."):
                    for idx, file in enumerate(uploaded_files, 1):
                        display_progress(idx, len(uploaded_files), f"Processing: {file.name}")
                        path, bytes_data = save_upload_and_cache_bytes(file)
                        try:
                            structured = run_async(st.session_state.processor.process_resume(str(path)))
                            cand_id = st.session_state.db.import_resume_data(structured)
                            results.append({
                                "name": file.name,
                                "bytes": bytes_data,
                                "candidate_id": cand_id,
                                "data": structured,
                                "is_pdf": file.name.lower().endswith(".pdf"),
                            })
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {e}")
                        finally:
                            path.unlink(missing_ok=True)  # Clean up after processing

                st.success("Processing complete!")
                st.subheader("Results")
                for res in results:
                    with st.expander(f"{res['name']} (ID: {res['candidate_id']})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Preview")
                            if res["is_pdf"]:
                                embed_pdf_from_bytes(res["bytes"])
                            else:
                                img = Image.open(BytesIO(res["bytes"]))
                                st.image(img, use_column_width=True)
                        with col2:
                            st.subheader("Extracted Data")
                            st.json(res["data"])
                            export_json(res["data"], "⬇️ Export JSON", f"{res['candidate_id']}.json")

        else:
            st.warning("No files uploaded yet.")

    with tab2:
        st.header("Google Drive Sync")
        stats = st.session_state.drive_processor.get_drive_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Processed", stats["total_processed_files"])
        col2.metric("Last 7 Days", stats["files_processed_last_week"])
        col3.metric("Status", "Active" if stats["configured"] else "Inactive")

        if not stats["configured"]:
            st.error("Please configure Google Drive in settings.")
        else:
            if st.button("Sync Now"):
                with st.spinner("Syncing..."):
                    new_files = run_async(st.session_state.drive_processor.process_new_files())
                    if new_files:
                        st.success(f"Processed {len(new_files)} new files")
                        for file in new_files:
                            st.markdown(f"""
                                <div class="card">
                                    <h5>{file['file_name']}</h5>
                                    <p>ID: {file['candidate_id']}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No new files.")

elif page == "🔍 Search & Match":
    tab1, tab2 = st.tabs(["🔎 Search Candidates", "🤖 Job Matching"])

    with tab1:
        st.header("Search Candidates")
        st.info("Search by skills, experience, or other criteria. Results are ranked by relevance.", icon="💡")
        
        col1, col2 = st.columns(2)
        with col1:
            skill = st.text_input("Skill Keyword", placeholder="e.g., Python, SQL", help="Search for specific skills")
        with col2:
            min_experience = st.number_input("Min Years Experience", min_value=0, value=0)
        
        if st.button("Search", type="primary"):
            with st.spinner("Searching..."):
                # Assuming db has an advanced search method; adjust as needed
                rows = st.session_state.db.search_candidates_by_skill(skill)  # Enhance this method if needed
            if rows:
                st.success(f"Found {len(rows)} candidates")
                for cid, name, email, phone, desc in rows:
                    st.markdown(f"""
                        <div class="card">
                            <h4>{name} (ID: {cid})</h4>
                            <p class="highlight">Summary: {desc or 'No summary available'}</p>
                            <p class="muted">Email: {email} | Phone: {phone}</p>
                            <div class="insight-card">Quick Insight: Strong match for queried skills</div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No candidates found. Try broadening your search.")

    with tab2:
        st.header("AI Job Matching")
        st.info("Enter job details for AI-powered matching. Highlights include match scores and gap analysis.", icon="💡")
        
        job_desc = st.text_area("Job Description", height=200, placeholder="Describe the role, required skills, experience...")
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Number of top candidates", 1, 20, 5)
        with col2:
            min_score = st.slider("Minimum Match Score", 0.0, 1.0, 0.5)
        
        if st.button("Match Candidates", type="primary"):
            with st.spinner("Analyzing and matching..."):
                try:
                    matches = run_async(st.session_state.job_matcher.find_best_candidates(job_desc, top_n))
                    filtered_matches = [m for m in matches if m[1].match_score >= min_score]
                    if filtered_matches:
                        st.success(f"Found {len(filtered_matches)} matching candidates above {min_score*100:.0f}% score")
                        st.markdown("---")
                        st.subheader("📊 Candidate Pool Analysis")
                        
                        gap_insights = st.session_state.job_matcher.get_gap_insights(filtered_matches)
                        for insight in gap_insights:
                            if insight.startswith("**"):
                                st.markdown(insight)
                            elif insight.strip() == "":
                                st.write("")
                            else:
                                st.write(insight)
                        
                        # Optional: Show detailed gap statistics
                        with st.expander("🔍 Detailed Gap Statistics"):
                            gap_summary = st.session_state.job_matcher.analyze_candidate_gaps(filtered_matches)
                            st.json(gap_summary.model_dump())
                        
                        for i, (cand, match) in enumerate(filtered_matches, 1):
                            st.markdown(f"""
                                <div class="card">
                                    <h4>#{i} {cand['name']} <span class="highlight">Score: {match.match_score:.0%}</span></h4>
                                    <p class="muted">Email: {cand['email']} | Phone: {cand['phone']}</p>
                                    <div>Skills: {', '.join(cand['skills'][:5]) + ('...' if len(cand['skills']) > 5 else '')}</div>
                                    <div class="progress-bar"><div class="progress-fill" style="width:{match.match_score * 100}%"></div></div>
                                    <h5>Reasoning</h5>
                                    <p>{match.reasoning}</p>
                                    <h5>Strengths</h5>
                                    {' '.join([f'<span class="tag">{s}</span>' for s in match.strengths])}
                                    <h5>Gaps</h5>
                                    {' '.join([f'<span class="warning-tag">{g}</span>' for g in match.gaps])}
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("No candidates meet the criteria.")
                except Exception as e:
                    st.error(f"Error: {e}")

elif page == "📊 Dashboard":
    st.header("System Dashboard")
    st.info("Key insights into your candidate pool. Hover over charts for details.", icon="💡")
    
    # Top Skills
    top_skills = st.session_state.db.conn.execute("SELECT sm.skill_name AS skill, COUNT(*) AS cnt FROM skills_master sm JOIN candidate_skills cs ON sm.skill_id = cs.skill_id GROUP BY sm.skill_name ORDER BY cnt DESC LIMIT 10").fetchdf()

    # Overall Insights Summary
    st.markdown("### Key Insights")
    total_candidates = stats.get("total_candidates", 0)
    avg_skills = stats.get("total_skills", 0) / total_candidates if total_candidates > 0 else 0
    st.markdown(f"<div class='insight-card'>Total Candidates: <b>{total_candidates}</b> | Average Skills per Candidate: <b>{avg_skills:.1f}</b></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='insight-card'>Most Common Skill: <b>{top_skills.iloc[0]['skill'] if not top_skills.empty else 'N/A'}</b> (Appears in {top_skills.iloc[0]['cnt'] if not top_skills.empty else 0} resumes)</div>", unsafe_allow_html=True)

    
    
    if not top_skills.empty:
        st.subheader("Top 10 Skills Distribution")
        skills_chart = alt.Chart(top_skills).mark_bar().encode(
            x=alt.X('cnt:Q', title='Count'),
            y=alt.Y('skill:N', sort='-x', title='Skill'),
            color=alt.Color('cnt:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['skill', 'cnt']
        ).properties(height=300)
        st.altair_chart(skills_chart, use_container_width=True)
    else:
        st.info("No skills data yet.")

    # Graduation Years
    grad = st.session_state.db.conn.execute("SELECT graduation_year AS year, COUNT(*) AS cnt FROM education WHERE graduation_year <> '' GROUP BY graduation_year ORDER BY year").fetchdf()
    if not grad.empty:
        st.subheader("Graduation Year Distribution")
        grad_chart = alt.Chart(grad).mark_bar().encode(
            x='year:N',
            y='cnt:Q',
            tooltip=['year', 'cnt']
        ).properties(height=300)
        st.altair_chart(grad_chart, use_container_width=True)
    else:
        st.info("No graduation data yet.")




    # Candidate Inflow
    inflow = st.session_state.db.conn.execute("SELECT strftime('%Y-%m', created_at) AS month, COUNT(*) AS cnt FROM candidates GROUP BY month ORDER BY month").fetchdf()
    if not inflow.empty:
        st.subheader("Candidate Inflow Over Time")
        inflow_chart = alt.Chart(inflow).mark_line(point=True).encode(
            x='month:T',
            y='cnt:Q',
            tooltip=['month', 'cnt']
        ).interactive()
        st.altair_chart(inflow_chart, use_container_width=True)
    else:
        st.info("No candidate inflow data yet.")


# Footer
st.markdown("<div class='footer'>© 2025 HR System</div>", unsafe_allow_html=True)
