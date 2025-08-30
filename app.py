"""
app.py – complete Streamlit front-end
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import logging
import nest_asyncio
import os
import threading
from pathlib import Path
from typing import List

import pandas as pd            # charts
from PIL import Image
import streamlit as st

from cv_processor import CVProcessor
from database import ResumeDatabase
from job_matcher import JobMatcher
from google_drive_processor import GoogleDriveProcessor
from config import get_settings  # ADD THIS LINE

# Initialize settings
settings = get_settings()  # ADD THIS LINE

# ═════════════════════════════════ APP CONFIG ═══════════════════════════════════
st.set_page_config(
    page_title="HR Resume Processing System",
    page_icon="📄",
    layout="wide",
)

nest_asyncio.apply()  # Allow nested asyncio.run in Streamlit's event loop

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ═════════════════════════ SESSION SINGLETONS ═══════════════════════════════════
if "db" not in st.session_state:
    st.session_state.db = ResumeDatabase("hr_resume_system.db")

if "processor" not in st.session_state:
    st.session_state.processor = CVProcessor()

if "job_matcher" not in st.session_state:
    st.session_state.job_matcher = JobMatcher(st.session_state.db)
# Add this to session state initialization
if "drive_processor" not in st.session_state:
    st.session_state.drive_processor = GoogleDriveProcessor(
        st.session_state.processor, 
        st.session_state.db
    )
# ════════════════════════════ HELPERS ═══════════════════════════════════════════
def _save_uploaded_file(uploaded) -> Path:
    """Persist an uploaded file to ./temp_uploads and return Path."""
    dest = Path("temp_uploads") / uploaded.name
    dest.parent.mkdir(exist_ok=True)
    with open(dest, "wb") as fh:
        fh.write(uploaded.getbuffer())
    return dest

def _load_png(path: str | Path) -> Image.Image | None:
    p = Path(path)
    return Image.open(p) if p.is_file() else None

def _embed_pdf(pdf_path: Path, height: int = 600) -> None:
    """Display a PDF in-line using <iframe>."""
    try:
        b64 = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
        html = (
            f"<iframe src='data:application/pdf;base64,{b64}' "
            f"width='100%' height='{height}px' type='application/pdf'></iframe>"
        )
        st.markdown(html, unsafe_allow_html=True)
    except Exception as exc:
        st.warning(f"Cannot preview PDF ({pdf_path.name}): {exc}")

def run_async(coroutine):
    """Run an async coroutine synchronously, even inside Streamlit's event loop."""
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # Run in a separate thread to avoid blocking the main loop
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

# ════════════════════════════ MAIN UI ═══════════════════════════════════════════
st.title("HR Resume Processing System")
st.markdown(
    "Upload CVs (PDF or image) to extract structured data automatically and store "
    "everything in the local DuckDB database."
)

# Navigation - single definition
page = st.sidebar.radio(
    "Navigation",
    ["Upload & Process", "Search Candidates", "Job Matching", "Google Drive Sync", "Database Stats"],
    help="Select a section",
)

# ───────────────────────────── Upload & Process ────────────────────────────────
if page == "Upload & Process":
    st.header("Upload and Process New Résumés")

    # optional workflow diagram
    with st.expander("Show CV-processing workflow diagram"):
        wf = _load_png("images/cv_processing_workflow.png")
        if wf:
            st.image(wf, caption="LangGraph workflow", use_container_width=False)
        else:
            st.info("Diagram not found (images/cv_processing_workflow.png)")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF or image files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    # batch processing ----------------------------------------------------------
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) queued for processing.")

        if st.button("Process Résumés"):
            status = st.empty()
            bar = st.progress(0)
            results: List[dict] = []

            for idx, upl in enumerate(uploaded_files, start=1):
                status.text(f"Processing {idx}/{len(uploaded_files)} — {upl.name}")
                tmp_path = _save_uploaded_file(upl)

                # optional warning for image files
                if tmp_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    st.warning(
                        f"Image '{upl.name}' treated as PDF – "
                        "OCR for images is limited."
                    )

                # main pipeline (run async function synchronously)
                try:
                    structured = run_async(st.session_state.processor.process_resume(str(tmp_path)))
                    cand_id = st.session_state.db.import_resume_data(structured)
                    results.append(
                        {
                            "path": tmp_path,
                            "name": upl.name,
                            "candidate_id": cand_id,
                            "data": structured,
                        }
                    )
                except Exception as exc:
                    st.error(f"❌ {upl.name}: {exc}")
                finally:
                    bar.progress(idx / len(uploaded_files))

            status.text("Batch complete ✅")

            # show interactive validation panels --------------------------------
            st.subheader("Validation")
            if results:
                for res in results:
                    with st.expander(f"{res['name']}  →  Candidate ID {res['candidate_id']}"):
                        col_pdf, col_json = st.columns(2)
                        with col_pdf:
                            st.markdown("##### Original CV")
                            if res["path"].suffix.lower() == ".pdf":
                                _embed_pdf(res["path"])
                            else:  # image preview
                                img = _load_png(res["path"])
                                if img:
                                    st.image(img, use_container_width=True)
                                else:
                                    st.info("Preview unavailable.")
                        with col_json:
                            st.markdown("##### Extracted Data")
                            st.json(res["data"], expanded=False)

            # cleanup temporary files
            for res in results:
                try:
                    res["path"].unlink(missing_ok=True)
                except Exception:
                    pass

    else:
        st.info("Please upload CV files to begin.")

# ───────────────────────────── Search Candidates ───────────────────────────────
elif page == "Search Candidates":
    st.header("Search Candidates by Skill")
    skill = st.text_input("Skill keyword (e.g. *Python*, *SQL*)")

    if st.button("Search") and skill:
        with st.spinner("Searching…"):
            rows = st.session_state.db.search_candidates_by_skill(skill)
        if rows:
            st.success(f"Found {len(rows)} candidate(s) for '{skill}'")
            for cid, name, email, phone, desc in rows:
                st.markdown(f"**Name:** {name}")
                st.markdown(f"**Email:** {email}")
                st.markdown(f"**Phone:** {phone}")
                st.markdown(f"**Summary:** {desc or '—'}")
                st.divider()
        else:
            st.warning("No matches.")

# ───────────────────────────── Job Matching ─────────────────────────────────────
elif page == "Job Matching":
    st.header("AI-Powered Job Matching")
    st.markdown("Enter a job description to find the best matching candidates with detailed analysis.")
    
    # Job description input
    job_description = st.text_area(
        "Job Description",
        height=200,
        placeholder="Enter the complete job description including required skills, experience level, responsibilities, etc.",
        help="Provide a detailed job description for accurate candidate matching"
    )
    
    # Number of candidates to show
    top_n = st.slider("Number of top candidates to show", min_value=1, max_value=10, value=5)
    
    if st.button("Find Best Candidates", type="primary") and job_description.strip():
        with st.spinner("Analyzing job requirements and evaluating candidates..."):
            try:
                # Run the async matching function
                matches = run_async(st.session_state.job_matcher.find_best_candidates(job_description, top_n))
                
                if matches:
                    st.success(f"Found {len(matches)} candidates. Results ranked by match score:")
                    
                    # Display results
                    for i, (candidate_details, match_result) in enumerate(matches, 1):
                        with st.expander(
                            f"#{i} - {candidate_details['name']} "
                            f"(Match: {match_result.match_score:.1%})",
                            expanded=i <= 3  # Expand top 3 by default
                        ):
                            # Create columns for better layout
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.markdown("##### Match Analysis")
                                st.markdown(f"**Overall Match Score:** {match_result.match_score:.1%}")
                                st.markdown(f"**Skills Match:** {match_result.skill_match_score:.1%}")
                                st.markdown(f"**Experience Match:** {match_result.experience_match_score:.1%}")
                                
                                st.markdown("##### Reasoning")
                                st.markdown(match_result.reasoning)
                                
                                if match_result.strengths:
                                    st.markdown("##### Key Strengths")
                                    for strength in match_result.strengths:
                                        st.markdown(f"✅ {strength}")
                                
                                if match_result.gaps:
                                    st.markdown("##### Potential Development Areas")
                                    for gap in match_result.gaps:
                                        st.markdown(f"⚠️ {gap}")
                            
                            with col2:
                                st.markdown("##### Candidate Details")
                                st.markdown(f"**Email:** {candidate_details['email']}")
                                st.markdown(f"**Phone:** {candidate_details['phone']}")
                                
                                if candidate_details['skills']:
                                    st.markdown("**Skills:**")
                                    skills_text = ", ".join(candidate_details['skills'][:10])  # Show first 10 skills
                                    if len(candidate_details['skills']) > 10:
                                        skills_text += f" (+{len(candidate_details['skills']) - 10} more)"
                                    st.markdown(skills_text)
                                
                                if candidate_details['description']:
                                    st.markdown("**Summary:**")
                                    st.markdown(candidate_details['description'][:200] + "..." if len(candidate_details['description']) > 200 else candidate_details['description'])
                
                else:
                    st.warning("No candidates found in the database.")
                    
            except Exception as e:
                st.error(f"An error occurred during matching: {e}")
                logger.error(f"Job matching error: {e}")
    
    elif not job_description.strip():
        st.info("Please enter a job description to begin matching.")
# ───────────────────────────── Google Drive Sync ──────────────────────────────────

# Add this new section after Job Matching section
elif page == "Google Drive Sync":
    st.header("🔄 Google Drive CV Processing")
    st.markdown("Automatically process new CVs from your Google Drive folder.")
    
    # Configuration status
    drive_stats = st.session_state.drive_processor.get_drive_stats()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Processed", drive_stats["total_processed_files"])
    col2.metric("Last 7 Days", drive_stats["files_processed_last_week"])
    col3.metric("Status", "✅ Configured" if drive_stats["configured"] else "❌ Not Configured")
    
    if not drive_stats["configured"]:
        st.error("⚠️ Google Drive is not properly configured!")
        st.markdown("""
        **Setup Instructions:**
        1. Add your `GOOGLE_DRIVE_API_KEY` to your `.env` file
        2. Add your `GOOGLE_DRIVE_FOLDER_ID` to your `.env` file
        3. Restart the application
        """)
        st.info("💡 **How to get Google Drive API Key:** Visit [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials")
    else:
        st.success("✅ Google Drive integration is configured and ready!")
        
        # Folder info
        with st.expander("📁 Folder Configuration"):
            st.code(f"Folder ID: {settings.GOOGLE_DRIVE_FOLDER_ID}")
            st.markdown(f"**Folder URL:** https://drive.google.com/drive/folders/{settings.GOOGLE_DRIVE_FOLDER_ID}")
        
        # Manual sync button
        # col1, col2 = st.columns([1, 3])
        
        # with col1:
        if st.button("🔄 Check for New CVs", type="primary"):
            with st.spinner("Checking Google Drive for new PDF files..."):
                try:
                    new_files = run_async(st.session_state.drive_processor.process_new_files())
                    
                    if new_files:
                        st.success(f"✅ Successfully processed {len(new_files)} new CV(s)!")
                        
                        # Show results
                        st.subheader("📋 Processing Results")
                        for result in new_files:
                            with st.expander(f"✅ {result['file_name']} → Candidate ID {result['candidate_id']}"):
                                col_info, col_data = st.columns(2)
                                
                                with col_info:
                                    st.markdown("**File Info:**")
                                    st.markdown(f"- **Name:** {result['file_name']}")
                                    st.markdown(f"- **Drive File ID:** {result['file_id']}")
                                    st.markdown(f"- **Candidate ID:** {result['candidate_id']}")
                                
                                with col_data:
                                    st.markdown("**Extracted Data:**")
                                    st.json(result['data'], expanded=False)
                    else:
                        st.info("ℹ️ No new CV files found in the Google Drive folder.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing Google Drive files: {e}")
                    logger.error(f"Google Drive sync error: {e}")
    
        # with col2:
        #     st.markdown("**💡 Tips:**")
        #     st.markdown("""
        #     - Drop PDF CVs into your configured Google Drive folder
        #     - Click 'Check for New CVs' to process them
        #     - Processed files are automatically tracked to avoid duplicates
        #     - Use the 'Job Matching' feature to find candidates for specific roles
        #     """)
        
        # Auto-sync option (future enhancement)
        st.markdown("---")
        with st.expander("🔮 Future Features"):
            st.markdown("""
            **Coming Soon:**
            - ⏰ Automatic periodic sync (every 15 minutes)
            - 📧 Email notifications when new CVs are processed
            - 📊 Processing history and logs
            - 🔍 Support for additional file formats (DOCX, etc.)
            """)


# ───────────────────────────── Database Stats ──────────────────────────────────
elif page == "Database Stats":
    st.header("Database Dashboard")

    # ER diagram
    schema = _load_png("images/database_schema.png")
    if schema:
        st.image(schema, caption="ER-diagram", use_container_width=False)

    # KPIs
    stats = st.session_state.db.get_database_stats()
    cols = st.columns(5)
    cols[0].metric("Candidates", stats["total_candidates"])
    cols[1].metric("Work Exps", stats["total_work_experiences"])
    cols[2].metric("Educations", stats["total_educations"])
    cols[3].metric("Skills", stats["total_skills"])
    cols[4].metric("Projects", stats["total_projects"])

    st.markdown("---")

    # charts --------------------------------------------------------------------
    conn = st.session_state.db.conn

    # résumé inflow
    inflow = conn.execute(
        """
        SELECT strftime('%Y-%m', created_at) AS month, COUNT(*) AS cnt
        FROM candidates
        GROUP BY month ORDER BY month
        """
    ).fetchdf()
    if not inflow.empty:
        st.subheader("Monthly candidate inflow")
        st.line_chart(inflow.set_index("month"))

    # top skills
    top_skills = conn.execute(
        """
        SELECT sm.skill_name AS skill, COUNT(*) AS cnt
        FROM skills_master sm
        JOIN candidate_skills cs ON sm.skill_id = cs.skill_id
        GROUP BY sm.skill_name ORDER BY cnt DESC LIMIT 10
        """
    ).fetchdf()
    if not top_skills.empty:
        st.subheader("Top 10 skills in current database")
        st.bar_chart(top_skills.set_index("skill").sort_values("cnt"))

    # grad-year distribution
    grad = conn.execute(
        """
        SELECT graduation_year AS year, COUNT(*) AS cnt
        FROM education
        WHERE graduation_year <> ''
        GROUP BY graduation_year ORDER BY year
        """
    ).fetchdf()
    if not grad.empty:
        st.subheader("Graduation year distribution")
        st.bar_chart(grad.set_index("year"))
