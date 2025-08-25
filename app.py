import streamlit as st
from pathlib import Path
import os
from cv_processor import CVProcessor
from database import ResumeDatabase
from PIL import Image

# --- App Configuration ---
st.set_page_config(
    page_title="HR Resume Processing System",
    page_icon="📄",
    layout="wide",
)

# --- Initialize Session State ---
if 'db' not in st.session_state:
    st.session_state.db = ResumeDatabase("hr_resume_system.db")
if 'processor' not in st.session_state:
    st.session_state.processor = CVProcessor()

# --- Helper Functions ---
def save_uploaded_file(uploaded_file):
    """Saves the uploaded file to a temporary directory and returns the path."""
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- UI Layout ---
st.title("HR Resume Processing System")
st.markdown("Upload a CV (PDF or Image) to extract information and store it in the database.")

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Process", "Search Candidates", "Database Stats"])

if page == "Upload & Process":
    st.header("Upload and Process a New Resume")
    uploaded_file = st.file_uploader("Choose a PDF or image file", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        st.info(f"File '{uploaded_file.name}' uploaded successfully.")

        if st.button("Process Resume"):
            with st.spinner("Processing... This may take a moment."):
                try:
                    # For image files, we would need to add logic here to convert them to PDF or use an OCR tool.
                    # For now, we assume the provided processor handles PDF.
                    if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        st.warning("Image processing is not fully implemented in the backend. Treating as PDF for now.")
                        # In a real-world scenario, you would convert the image to PDF here.

                    structured_data = st.session_state.processor.process_resume(str(file_path))
                    candidate_id = st.session_state.db.import_resume_data(structured_data)
                    
                    st.success(f"Resume processed successfully! Candidate ID: {candidate_id}")
                    st.subheader("Extracted Information")
                    st.json(structured_data)

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
        
        # Clean up the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

elif page == "Search Candidates":
    st.header("Search for Candidates by Skill")
    skill_query = st.text_input("Enter a skill to search for (e.g., Python, SQL)")

    if st.button("Search"):
        if skill_query:
            with st.spinner("Searching..."):
                results = st.session_state.db.search_candidates_by_skill(skill_query)
                if results:
                    st.success(f"Found {len(results)} candidate(s) with the skill '{skill_query}'")
                    for row in results:
                        st.write(f"**Name:** {row[1]}")
                        st.write(f"**Email:** {row[2]}")
                        st.write(f"**Phone:** {row[3]}")
                        st.divider()
                else:
                    st.warning(f"No candidates found with the skill '{skill_query}'")
        else:
            st.warning("Please enter a skill to search.")

elif page == "Database Stats":
    st.header("Database Statistics")
    if st.button("Refresh Stats"):
        with st.spinner("Fetching stats..."):
            stats = st.session_state.db.get_database_stats()
            st.write(f"**Total Candidates:** {stats['total_candidates']}")
            st.write(f"**Total Work Experiences:** {stats['total_work_experiences']}")
            st.write(f"**Total Education Records:** {stats['total_educations']}")
            st.write(f"**Total Unique Skills:** {stats['total_skills']}")
            st.write(f"**Total Projects:** {stats['total_projects']}")

