# AI HR Resume Processing System

A streamlined, end-to-end HR assistant that ingests candidate resumes (PDF or image), extracts structured information using Azure services, and stores it in a searchable DuckDB talent database via a friendly Streamlit UI. This project is built for fast candidate intake, search, and reporting workflows.

## Highlights

- Upload CVs (PDF or image) and extract structured data with Azure Document Intelligence and Azure OpenAI.
- Normalize dates, handle “Present/Current,” and persist into a snowflake-style DuckDB schema.
- Friendly Streamlit UI with tabs, status indicators, toast notifications, and a custom theme.


## Features

- Resume ingestion: PDF and common image types.
- OCR + LLM pipeline: layout-aware text extraction and schema-constrained structuring.
- Data persistence: candidates, experience, education, skills, and projects.
- Talent search: filter candidates by skill.
- Stats dashboard: quick database metrics.
- Environment-driven config: .env + typed settings loader.


## Architecture

- UI: Streamlit app (tabs, progress/status, toast).
- Processing: LangGraph workflow invoking Azure Document Intelligence and Azure OpenAI.
- Data Model: DuckDB snowflake schema with sequences and M:N skills.
- Config: Typed settings via Pydantic Settings and .env loader.


## Project Structure

```
AI_hr_system/
├── app.py                  # Main entry point (handles home page and shared code)
├── pages/                  # Folder for page views (Streamlit auto-detects this)
│   ├── upload_and_sync.py  # Code for "Upload & Sync" page
│   ├── search_and_match.py # Code for "Search & Match" page
│   └── dashboard.py        # Code for "Dashboard" page
├── utils.py                # New file for shared helpers, imports, and session state
├── config.py               # (Assuming this exists; otherwise, keep in utils.py)
├── cv_processor.py         # Existing backend files (unchanged)
├── database.py
├── job_matcher.py
├── google_drive_processor.py
└── hr_resume_system.db     # Database file

```


## Requirements

- Python 3.10+
- Azure Document Intelligence resource (endpoint + key)
- Azure OpenAI resource (endpoint + key + deployment name)


## Installation

1) Create and activate a virtual environment.
2) Install dependencies.
3) Prepare environment variables.

Example:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Minimal requirements.txt:

```
streamlit
pandas
duckdb
python-dotenv
pydantic
pydantic-settings
azure-ai-documentintelligence
azure-core
langchain-openai
langgraph
pillow
python-dateutil
```


## Configuration

All credentials and settings are read from .env via config.py (typed and validated). Copy the example and fill in values:

.env.example

```
ENV=development
LOG_LEVEL=INFO

# Azure Document Intelligence
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=https://<your-docintel>.cognitiveservices.azure.com/
AZURE_DOCUMENT_INTELLIGENCE_KEY=<your-docintel-key>

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<your-aoai-endpoint>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-aoai-key>
AZURE_OPENAI_DEPLOYMENT=gpt-4.1-mini
AZURE_OPENAI_API_VERSION=2025-04-01-preview

# Database
DUCKDB_PATH=hr_resume_system.db
```

config.py (usage example)

```python
from config import get_settings

settings = get_settings()
print(settings.DUCKDB_PATH)
```

Notes

- Environment variables override .env values at runtime.
- Create variants like .env.development or .env.production and set ENV accordingly.


## Theming (optional but recommended)

Customize the app with a theme file:

.streamlit/config.toml

```
[theme]
primaryColor = "#3B82F6"
backgroundColor = "#0B1220"
secondaryBackgroundColor = "#111827"
textColor = "#E5E7EB"
font = "sans serif"
```


## Running the App

```bash
streamlit run app.py
```

- Upload a PDF or image file in the “Upload \& Process” tab.
- After processing, the app saves the extracted data and shows JSON results.
- Use “Search” to find candidates by skill (e.g., SQL, Python).
- Use “Stats” for database metrics.


## Database Schema

- candidates(candidate_id, name, email, phone, created_at)
- work_experience(work_exp_id, candidate_id, job_title, company, start_date, end_date, description)
- education(education_id, candidate_id, degree, institution, graduation_year)
- skills_master(skill_id, skill_name)
- candidate_skills(candidate_id, skill_id)
- projects(project_id, candidate_id, project_name, description)

Conventions

- Dates are stored as DATE or NULL; ongoing roles have end_date = NULL.
- Skills are normalized (master + junction).


## Key Modules

- app.py: Streamlit UI with tabs, status, progress, toast, and a clean layout.
- cv_processor.py: Azure Document Intelligence + Azure OpenAI pipeline, with LangGraph state steps.
- database.py: Snowflake schema creation, robust import with date normalization, candidate search, and stats.
- config.py: Pydantic Settings + dotenv loader with environment precedence.


## Date Normalization

- “Present/Current/Now/Ongoing/-” → NULL.
- YYYY → YYYY-01-01, YYYY-MM → YYYY-MM-01, or parsed into ISO (YYYY-MM-DD).
- Invalid/unparseable values → NULL to avoid insert failures.


## Troubleshooting

- begin_analyze_document signature: pass the file stream (file object) instead of raw bytes.
    - Correct: body=f
    - Avoid: body=f.read()
- “Conversion Error: invalid date field format: ‘Present’…”
    - Ensure the date normalizer maps “Present/Current” to NULL and returns ISO dates otherwise.
- UI shows no results after insert
    - Verify DUCKDB_PATH, confirm the .db file is being written, and check stats for row counts.
- LLM errors or timeouts
    - Check API keys, deployment name, and API version; verify the region matches the deployment.


## Security

- Do not commit .env or any secret keys.
- Restrict access to the DuckDB file on shared systems.
- Rotate keys regularly and prefer environment variables in production deployments.


## Development Tips

- Use a small temp_uploads/ folder for uploads; clean files after processing.
- Keep the extraction prompt strict to minimize hallucinations and enforce schema.
- Add more validators if extending the schema (e.g., phone/email formats).


## Roadmap

- Bulk upload (multiple resumes).
- Candidate profiles view with timeline and attachments.
- Role-fit scoring with semantic retrieval and ranking.
- Export reports (CSV/JSON) and admin audit logs.


## Testing

- Add unit tests for _to_iso_date_or_none and data ingestion.
- Use controlled sample resumes (PDF + images) to validate OCR/LLM behavior.
- Mock external services for CI runs.


## Screenshot

Place screenshots in assets/ and reference them here once available.

```
![App Screenshot](assets/screenshot.png)
```


## License

Add a suitable license file in the project root if distributing.

## Acknowledgements

Thanks to the open-source community and platform SDKs that make streamlined HR data processing practical.
<span style="display:none">[^1]</span>

<div style="text-align: center">⁂</div>

[^1]: image.jpg

