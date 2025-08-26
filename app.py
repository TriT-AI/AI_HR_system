import os
from pathlib import Path
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

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Process", "Search Candidates", "Database Stats"])

# ── Page: Upload & Process ───────────────────────────────────────────────────────
if page == "Upload & Process":
    st.header("Upload and Process a New Resume")

    # Collapsible workflow diagram
    with st.expander("Show CV-processing workflow diagram"):
        wf_img = _load_png("images/cv_processing_workflow.png")
        if wf_img:
            st.image(wf_img, use_container_width =False, caption="LangGraph CV-processing workflow")
        else:
            st.info("Workflow diagram not found (images/cv_processing_workflow.png)")

    uploaded_file = st.file_uploader(
        "Choose a PDF or image file",
        type=["pdf", "png", "jpg", "jpeg"],
        label_visibility="visible",
    )

    if uploaded_file:
        file_path = save_uploaded_file(uploaded_file)
        st.info(f"File “{uploaded_file.name}” uploaded successfully.")

        if st.button("Process Resume"):
            with st.spinner("Processing …  this may take a moment."):
                try:
                    # NOTE: image-to-PDF conversion is not implemented; treat as PDF for now.
                    if file_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        st.warning("Image processing not fully implemented—treating image as PDF.")

                    structured = st.session_state.processor.process_resume(str(file_path))
                    cand_id = st.session_state.db.import_resume_data(structured)

                    st.success(f"Resume processed successfully 🎉   Candidate ID: {cand_id}")
                    st.subheader("Extracted information")
                    st.json(structured)

                except Exception as exc:
                    st.error(f"Processing failed: {exc}")

        # Clean-up
        if file_path.exists():
            os.remove(file_path)

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

    # Always show the ER diagram
    schema_img = _load_png("images/database_schema.png")
    if schema_img:
        st.image(schema_img, use_container_width=False, caption="Database ER-diagram")
    else:
        st.info("Database diagram not found (images/database_schema.png)")

    if st.button("Refresh Stats"):
        with st.spinner("Fetching stats …"):
            stats = st.session_state.db.get_database_stats()
            st.metric("Total Candidates", stats["total_candidates"])
            st.metric("Total Work Experiences", stats["total_work_experiences"])
            st.metric("Total Education Records", stats["total_educations"])
            st.metric("Total Unique Skills", stats["total_skills"])
            st.metric("Total Projects", stats["total_projects"])
