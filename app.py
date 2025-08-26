import os
from pathlib import Path

import pandas as pd
from PIL import Image
import streamlit as st

from cv_processor import CVProcessor
from database import ResumeDatabase


# ── App configuration ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Resume Processing System",
    page_icon="📄",
    layout="wide",
)

# ── Session-state singletons ─────────────────────────────────────────────────────
if "db" not in st.session_state:
    st.session_state.db = ResumeDatabase("hr_resume_system.db")

if "processor" not in st.session_state:
    st.session_state.processor = CVProcessor()


# ── Utilities ────────────────────────────────────────────────────────────────────
def save_uploaded_file(uploaded_file) -> Path:
    """Save an uploaded file to ./temp_uploads and return the path."""
    tmp_dir = Path("temp_uploads")
    tmp_dir.mkdir(exist_ok=True)
    file_path = tmp_dir / uploaded_file.name
    with open(file_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return file_path


def _load_png(path: str | Path) -> Image.Image | None:
    """Return a PIL Image if the file exists; otherwise None."""
    p = Path(path)
    return Image.open(p) if p.is_file() else None


# ── UI layout ────────────────────────────────────────────────────────────────────
st.title("HR Resume Processing System")
st.markdown("Upload a CV (PDF or image) to extract information and store it in the database.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Process", "Search Candidates", "Database Stats"])

# ── Page: Upload & Process ───────────────────────────────────────────────────────
if page == "Upload & Process":
    st.header("Upload and Process a New Resume")

    # Collapsible workflow diagram
    with st.expander("Show CV-processing workflow diagram"):
        wf_img = _load_png("images/cv_processing_workflow.png")
        if wf_img:
            st.image(wf_img, use_container_width=False, caption="LangGraph CV-processing workflow")
        else:
            st.info("Workflow diagram not found (images/cv_processing_workflow.png)")

    uploaded_files = st.file_uploader(
        "Choose one or more PDF or image files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="visible",
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) uploaded successfully.")

        if st.button("Process Resumes"):
            with st.spinner("Processing … this may take a moment."):
                try:
                    num_files = len(uploaded_files)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results = []

                    for idx, uploaded_file in enumerate(uploaded_files, 1):
                        status_text.text(f"Processing file {idx} of {num_files}: {uploaded_file.name}")

                        file_path = save_uploaded_file(uploaded_file)

                        # NOTE: image-to-PDF conversion is not implemented; treat as PDF for now.
                        if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                            st.warning(f"Image processing not fully implemented—treating {uploaded_file.name} as PDF.")

                        structured = st.session_state.processor.process_resume(str(file_path))
                        cand_id = st.session_state.db.import_resume_data(structured)

                        results.append({"file": uploaded_file.name, "candidate_id": cand_id})

                        progress_bar.progress(idx / num_files)

                        # Clean up this file
                        if file_path.exists():
                            os.remove(file_path)

                    status_text.text("All files processed.")
                    st.success(f"Processed {num_files} resumes successfully 🎉")

                    # Display results
                    st.subheader("Processing Results")
                    for res in results:
                        st.markdown(f"**File:** {res['file']} → **Candidate ID:** {res['candidate_id']}")

                except Exception as exc:
                    st.error(f"Processing failed: {exc}")

    else:
        st.info("Please upload one or more files to process.")

# ── Page: Search Candidates ─────────────────────────────────────────────────────
elif page == "Search Candidates":
    st.header("Search for Candidates by Skill")
    skill = st.text_input("Enter a skill (e.g., Python, SQL)")

    if st.button("Search"):
        if skill:
            with st.spinner("Searching …"):
                rows = st.session_state.db.search_candidates_by_skill(skill)
                if rows:
                    st.success(f"Found {len(rows)} candidate(s) with “{skill}”")
                    for row in rows:
                        st.markdown(f"**Name:** {row[1]}")
                        st.markdown(f"**Email:** {row[2]}")
                        st.markdown(f"**Phone:** {row[3]}")
                        st.markdown(f"**Candidate Description:** {row[4]}")
                        st.divider()
                else:
                    st.warning(f"No candidates found with “{skill}”.")
        else:
            st.warning("Please enter a skill to search.")

# ── Page: Database Stats ────────────────────────────────────────────────────────
elif page == "Database Stats":
    st.header("Database Statistics")

    # ER diagram
    schema_img = _load_png("images/database_schema.png")
    if schema_img:
        st.image(schema_img, use_container_width=False, caption="Database ER-diagram")
    else:
        st.info("Database diagram not found (images/database_schema.png)")

    # Numeric KPIs
    with st.spinner("Fetching stats …"):
        stats = st.session_state.db.get_database_stats()

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Candidates", stats["total_candidates"])
    kpi_cols[1].metric("Work Experiences", stats["total_work_experiences"])
    kpi_cols[2].metric("Education Records", stats["total_educations"])
    kpi_cols[3].metric("Unique Skills", stats["total_skills"])
    kpi_cols[4].metric("Projects", stats["total_projects"])

    st.markdown("---")

    # ── Insight 1: Head-count growth ───────────────────────────────────────────
    st.subheader("Candidate inflow over time")
    df_growth = st.session_state.db.conn.execute(
        """
        SELECT strftime('%Y-%m', created_at) AS month,
               COUNT(*)                        AS candidates
        FROM candidates
        GROUP BY month
        ORDER BY month
        """
    ).fetchdf()
    if not df_growth.empty:
        st.line_chart(df_growth.set_index("month"))
    else:
        st.info("No candidates yet – growth chart unavailable.")

    # ── Insight 2: Résumés volume by upload month ──────────────────────────────
    st.subheader("Résumés processed each month")
    st.bar_chart(df_growth.set_index("month"))  # reuse if already fetched

    st.markdown("---")

    # ── Insight 3: Most requested skills ──────────────────────────────────────
    st.subheader("Top 10 skills across all candidates")
    df_skills = st.session_state.db.conn.execute(
        """
        SELECT sm.skill_name AS skill, COUNT(*) AS cnt
        FROM skills_master sm
        JOIN candidate_skills cs ON sm.skill_id = cs.skill_id
        GROUP BY sm.skill_name
        ORDER BY cnt DESC
        LIMIT 10
        """
    ).fetchdf()
    if not df_skills.empty:
        st.bar_chart(df_skills.set_index("skill").sort_values("cnt"))
    else:
        st.info("No skills recorded yet – skill chart unavailable.")

    st.markdown("---")

    # ── Insight 4: Graduation-year distribution ───────────────────────────────
    st.subheader("Graduation year distribution")
    df_grad = st.session_state.db.conn.execute(
        """
        SELECT graduation_year AS year,
               COUNT(*)            AS cnt
        FROM education
        WHERE graduation_year <> ''
        GROUP BY graduation_year
        ORDER BY year
        """
    ).fetchdf()
    if not df_grad.empty:
        st.bar_chart(df_grad.set_index("year"))
    else:
        st.info("No graduation-year data yet – education chart unavailable.")
