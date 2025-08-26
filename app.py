"""
app.py – complete Streamlit front-end

Key additions
• Upload & Process page now accepts *multiple* CV files.
• After every batch run each résumé appears in its own expander with:
  – an embedded PDF preview (left)  
  – the extracted JSON (right) so recruiters can validate instantly.
• Progress bar and status text show batch progress.
• All other pages (search, dashboard with charts) unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
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


# ═════════════════════════════════ APP CONFIG ═══════════════════════════════════
st.set_page_config(
    page_title="HR Resume Processing System",
    page_icon="📄",
    layout="wide",
)

nest_asyncio.apply()  # Allow nested asyncio.run in Streamlit's event loop

# ═════════════════════════ SESSION SINGLETONS ═══════════════════════════════════
if "db" not in st.session_state:
    st.session_state.db = ResumeDatabase("hr_resume_system.db")

if "processor" not in st.session_state:
    st.session_state.processor = CVProcessor()


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

page = st.sidebar.radio(
    "Navigation",
    ["Upload & Process", "Search Candidates", "Database Stats"],
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
            st.success(f"Found {len(rows)} candidate(s) for “{skill}”")
            for cid, name, email, phone, desc in rows:
                st.markdown(f"**Name:** {name}")
                st.markdown(f"**Email:** {email}")
                st.markdown(f"**Phone:** {phone}")
                st.markdown(f"**Summary:** {desc or '—'}")
                st.divider()
        else:
            st.warning("No matches.")

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
