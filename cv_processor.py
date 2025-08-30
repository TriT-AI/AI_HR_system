from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional, Any, Dict, Required, NotRequired

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentSpan
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, SecretStr
from config import get_settings

settings = get_settings()

# Configure logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

# Azure clients
document_intelligence_client = DocumentIntelligenceClient(
    endpoint=settings.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT,
    credential=AzureKeyCredential(settings.AZURE_DOCUMENT_INTELLIGENCE_KEY),
)

llm = AzureChatOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

# Pydantic Models for Structured Output
class ContactInfo(BaseModel):
    email: str = Field(description="Candidate's email address", default="")
    phone: str = Field(description="Candidate's phone number", default="")

class EmploymentRecord(BaseModel):
    position: str = Field(description="Job title/position", default="")
    employer: str = Field(description="Company/employer name", default="")
    start_date: str = Field(description="Start date (YYYY-MM-DD format)", default="")
    end_date: str = Field(description="End date (YYYY-MM-DD or 'Present')", default="")
    summary: str = Field(description="Job description and responsibilities", default="")

class EducationRecord(BaseModel):
    institution: str = Field(description="Educational institution name", default="")
    degree: str = Field(description="Degree or field of study", default="")
    graduation_year: str = Field(description="Graduation year", default="")

class Project(BaseModel):
    name: str = Field(description="Project name", default="")
    description: str = Field(description="Project description", default="")

class Employment(BaseModel):
    history: List[EmploymentRecord] = Field(description="List of employment records", default_factory=list)

class Education(BaseModel):
    history: List[EducationRecord] = Field(description="List of education records", default_factory=list)

class Resume(BaseModel):
    name: str = Field(description="Full name of the candidate", default="")
    contact: ContactInfo = Field(description="Contact information", default_factory=ContactInfo)
    employment: Employment = Field(description="Employment history", default_factory=Employment)
    education: Education = Field(description="Education history", default_factory=Education)
    skills: List[str] = Field(description="List of skills and competencies", default_factory=list)
    projects: List[Project] = Field(description="List of notable projects", default_factory=list)
    candidate_description: str = Field(description="Concise summarization of the whole CV", default="")

# LangGraph State Definition (use PEP 655 Required/NotRequired to satisfy Pylance)
class ProcessingState(TypedDict):
    pdf_path: Required[str]
    extracted_text: Required[str]
    structured_data: NotRequired[Optional[dict]]
    error_message: NotRequired[Optional[str]]
    processing_status: Required[str]

# Prompt Template (updated to include summarization instruction)
EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert HR data extraction specialist. Extract information from the following resume text and structure it according to the provided schema.

Resume Text:
{resume_text}

Instructions:
- Extract all available information accurately
- For name, phone, email if not exist, make up a random one, be creative with the name
- Use empty strings for missing text fields
- Use empty arrays for missing list fields
- Format dates as YYYY-MM-DD when possible
- For current positions, use "Present" as end date
- Be precise and avoid hallucinations
- Generate a concise summarization of the whole CV in the 'candidate_description' field, highlighting key experiences, skills, and qualifications
"""
)

def _min_span_offset(spans: Optional[List[DocumentSpan]]) -> int:
    """
    Return the minimum character offset across spans, or 0 if spans is None/empty.
    """
    if not spans:
        return 0
    return min(s.offset for s in spans if hasattr(s, "offset"))

async def extract_text_from_pdf(state: ProcessingState) -> ProcessingState:
    """
    Extract text from PDF using Azure Document Intelligence.
    """
    try:
        logger.info("Starting Azure Document Intelligence extraction for: %s", state["pdf_path"])

        # Validate file exists
        pdf_path = Path(state["pdf_path"])
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Read PDF file and pass the file stream directly
        with open(pdf_path, "rb") as f:
            poller = await asyncio.to_thread(
                document_intelligence_client.begin_analyze_document,
                "prebuilt-layout",
                body=f,  # pass stream, not f.read()
            )
            result: AnalyzeResult = poller.result()

        # Extract text from the analysis result
        extracted_text = ""

        # Method 1: Extract from content property (preferred in newer versions)
        if getattr(result, "content", None):
            extracted_text = result.content
            logger.info("Extracted text from content property: %d characters", len(extracted_text))

        # Method 2: Extract from paragraphs (preserves document structure)
        elif getattr(result, "paragraphs", None):
            paragraphs = result.paragraphs or []
            logger.info("Found %d paragraphs", len(paragraphs))
            # Sort paragraphs by the minimum span offset (spans reference content offsets)
            sorted_paragraphs = sorted(paragraphs, key=lambda p: _min_span_offset(p.spans))
            for paragraph in sorted_paragraphs:
                if getattr(paragraph, "content", None):
                    extracted_text += paragraph.content + "\n"

        # Method 3: Fallback to pages if no paragraphs found
        elif getattr(result, "pages", None):
            pages = result.pages or []
            logger.info("Found %d pages, extracting from lines", len(pages))
            for page in pages:
                lines = getattr(page, "lines", None) or []
                for line in lines:
                    if getattr(line, "content", None):
                        extracted_text += line.content + "\n"

        if not extracted_text.strip():
            raise ValueError("No text could be extracted from the PDF using Azure Document Intelligence")

        logger.info("Successfully extracted %d characters using Azure Document Intelligence", len(extracted_text))

        return {
            **state,
            "extracted_text": extracted_text,
            "processing_status": "text_extracted",
        }

    except Exception as e:
        error_msg = f"Azure Document Intelligence extraction failed: {e}"
        logger.error("%s", error_msg)
        return {
            **state,
            "error_message": error_msg,
            "processing_status": "ocr_failed",
        }

async def structure_resume_data(state: ProcessingState) -> ProcessingState:
    """
    Use the LLM to structure the extracted resume text into the Resume schema.
    Handles both Pydantic BaseModel and dict return types.
    """
    try:
        logger.info("Starting LLM-based data structuring")
        structured_llm = llm.with_structured_output(Resume)
        extraction_chain = EXTRACTION_PROMPT | structured_llm
        structured_resume = await extraction_chain.ainvoke({"resume_text": state["extracted_text"]})

        if isinstance(structured_resume, BaseModel):
            structured_data: Dict[str, Any] = structured_resume.model_dump()
        else:
            # Accept dict-like as-is; ensure a plain dict for JSON serialization
            structured_data = dict(structured_resume)

        logger.info("Successfully structured resume data")
        return {**state, "structured_data": structured_data, "processing_status": "completed"}

    except Exception as e:
        error_msg = f"Data structuring failed: {e}"
        logger.error("%s", error_msg)
        return {**state, "error_message": error_msg, "processing_status": "structuring_failed"}

async def should_continue_processing(state: ProcessingState) -> str:
    if state.get("error_message"):
        return END
    status = state.get("processing_status", "")
    return "structure_data" if status == "text_extracted" else END

def create_cv_processing_workflow():
    workflow = StateGraph(ProcessingState)
    workflow.add_node("extract_text", extract_text_from_pdf)
    workflow.add_node("structure_data", structure_resume_data)
    workflow.set_entry_point("extract_text")
    workflow.add_conditional_edges(
        "extract_text",
        should_continue_processing,
        {"structure_data": "structure_data", END: END},
    )
    workflow.add_edge("structure_data", END)

    # Compile the graph first
    compiled = workflow.compile()

    # Ensure images directory exists and export Mermaid PNG
    try:
        images_dir = Path("images")
        images_dir.mkdir(parents=True, exist_ok=True)
        output_path = images_dir / "cv_processing_workflow.png"
        compiled.get_graph().draw_mermaid_png(output_file_path=str(output_path))
        logger.info("Workflow diagram saved to: %s", output_path)
    except Exception as viz_err:
        # Visualization is best-effort; keep going if it fails (e.g., no internet or missing deps)
        logger.warning("Could not render workflow diagram: %s", viz_err)

    return compiled

class CVProcessor:
    def __init__(self) -> None:
        self.workflow = create_cv_processing_workflow()

    async def process_resume(self, pdf_path: str, output_path: Optional[str] = None) -> dict:
        initial_state: ProcessingState = {
            "pdf_path": pdf_path,
            "extracted_text": "",
            "structured_data": None,
            "error_message": None,
            "processing_status": "initialized",
        }
        final_state = await self.workflow.ainvoke(initial_state)

        if final_state.get("error_message"):
            raise ValueError(final_state["error_message"])

        structured_data = final_state.get("structured_data", {}) or {}

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            logger.info("Structured data saved to: %s", output_path)

        return structured_data