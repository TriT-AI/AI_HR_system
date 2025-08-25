from typing import TypedDict, Annotated, List, Optional
import json
import logging
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field, SecretStr
from config import get_settings
settings = get_settings()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

# LangGraph State Definition
class ProcessingState(TypedDict):
    pdf_path: str
    extracted_text: Annotated[str, "Document Intelligence extracted text"]
    structured_data: Annotated[Optional[dict], "Parsed resume data"]
    error_message: Annotated[Optional[str], "Error information"]
    processing_status: Annotated[str, "Current processing status"]

# Prompt Template
EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert HR data extraction specialist. Extract information from the following resume text and structure it according to the provided schema.

Resume Text:
{resume_text}

Instructions:
- Extract all available information accurately
- Use empty strings for missing text fields
- Use empty arrays for missing list fields
- Format dates as YYYY-MM-DD when possible
- For current positions, use "Present" as end date
- Be precise and avoid hallucinations
"""
)

def extract_text_from_pdf(state: ProcessingState) -> ProcessingState:
    """
    Extract text from PDF using Azure Document Intelligence.
    
    Args:
        state: Current processing state
        
    Returns:
        Updated state with extracted text or error information
    """
    try:
        logger.info(f"Starting Azure Document Intelligence extraction for: {state['pdf_path']}")
        
        # Validate file exists
        pdf_path = Path(state["pdf_path"])
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Read PDF file and pass the file stream directly
        with open(pdf_path, "rb") as f:
            # Pass the file stream directly, not f.read()
            poller = document_intelligence_client.begin_analyze_document(
                "prebuilt-layout", 
                body=f  # Pass file stream, not f.read()
            )
            
            # Get the analysis result
            result: AnalyzeResult = poller.result()
        
        # Extract text from the analysis result
        extracted_text = ""
        
        # Method 1: Extract from content property (preferred in newer versions)
        if hasattr(result, 'content') and result.content:
            extracted_text = result.content
            logger.info(f"Extracted text from content property: {len(extracted_text)} characters")
        
        # Method 2: Extract from paragraphs (preserves document structure)
        elif result.paragraphs:
            logger.info(f"Found {len(result.paragraphs)} paragraphs")
            # Sort paragraphs by their position in the document
            sorted_paragraphs = sorted(
                result.paragraphs, 
                key=lambda p: (p.spans.offset if p.spans else 0)
            )
            
            for paragraph in sorted_paragraphs:
                if paragraph.content:
                    extracted_text += paragraph.content + "\n"
        
        # Method 3: Fallback to pages if no paragraphs found
        elif result.pages:
            logger.info(f"Found {len(result.pages)} pages, extracting from lines")
            for page in result.pages:
                if page.lines:
                    for line in page.lines:
                        extracted_text += line.content + "\n"
        
        if not extracted_text.strip():
            raise ValueError("No text could be extracted from the PDF using Azure Document Intelligence")
        
        logger.info(f"Successfully extracted {len(extracted_text)} characters using Azure Document Intelligence")
        
        return {
            **state,
            "extracted_text": extracted_text,
            "processing_status": "text_extracted"
        }
        
    except Exception as e:
        error_msg = f"Azure Document Intelligence extraction failed: {str(e)}"
        logger.error(error_msg)
        return {
            **state,
            "error_message": error_msg,
            "processing_status": "ocr_failed"
        }

def structure_resume_data(state: ProcessingState) -> ProcessingState:
    try:
        logger.info("Starting LLM-based data structuring")
        structured_llm = llm.with_structured_output(Resume)
        extraction_chain = EXTRACTION_PROMPT | structured_llm
        structured_resume = extraction_chain.invoke({"resume_text": state["extracted_text"]})
        structured_data = structured_resume.dict()
        logger.info("Successfully structured resume data")
        return {**state, "structured_data": structured_data, "processing_status": "completed"}
    except Exception as e:
        error_msg = f"Data structuring failed: {str(e)}"
        logger.error(error_msg)
        return {**state, "error_message": error_msg, "processing_status": "structuring_failed"}

def should_continue_processing(state: ProcessingState) -> str:
    if state.get("error_message"):
        return END
    status = state.get("processing_status", "")
    return "structure_data" if status == "text_extracted" else END

def create_cv_processing_workflow():
    workflow = StateGraph(ProcessingState)
    workflow.add_node("extract_text", extract_text_from_pdf)
    workflow.add_node("structure_data", structure_resume_data)
    workflow.set_entry_point("extract_text")
    workflow.add_conditional_edges("extract_text", should_continue_processing, {"structure_data": "structure_data", END: END})
    workflow.add_edge("structure_data", END)
    return workflow.compile()

class CVProcessor:
    def __init__(self):
        self.workflow = create_cv_processing_workflow()
    
    def process_resume(self, pdf_path: str, output_path: Optional[str] = None) -> dict:
        initial_state: ProcessingState = {
            "pdf_path": pdf_path,
            "extracted_text": "",
            "structured_data": None,
            "error_message": None,
            "processing_status": "initialized"
        }
        final_state = self.workflow.invoke(initial_state)
        
        if final_state.get("error_message"):
            raise ValueError(final_state["error_message"])
            
        structured_data = final_state.get("structured_data", {})
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Structured data saved to: {output_path}")
            
        return structured_data

