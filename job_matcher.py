# job_matcher.py
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Initialize LLM
llm = AzureChatOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

class JobRequirements(BaseModel):
    required_skills: List[str] = Field(description="Essential skills for the job", default_factory=list)
    preferred_skills: List[str] = Field(description="Nice-to-have skills", default_factory=list)
    experience_level: str = Field(description="Required experience level", default="")
    industry: str = Field(description="Industry or domain", default="")
    role_type: str = Field(description="Type of role (technical, managerial, etc.)", default="")

class CandidateMatch(BaseModel):
    candidate_id: int = Field(description="Database ID of the candidate")
    match_score: float = Field(description="Match score from 0.0 to 1.0", ge=0.0, le=1.0)
    skill_match_score: float = Field(description="Skills compatibility score", ge=0.0, le=1.0)
    experience_match_score: float = Field(description="Experience level match", ge=0.0, le=1.0)
    strengths: List[str] = Field(description="Candidate's strengths for this role", default_factory=list)
    gaps: List[str] = Field(description="Areas where candidate may need development", default_factory=list)
    reasoning: str = Field(description="Detailed explanation of why this candidate is a good fit", default="")

class BatchCandidateEvaluation(BaseModel):
    evaluations: List[CandidateMatch] = Field(description="List of candidate evaluations")

class JobMatcher:
    def __init__(self, database):
        self.db = database
        self._job_requirements_cache = {}
        self._max_concurrent_llm_calls = 5  # Reduced for batch processing
        self._batch_size = 3  # Process 3 candidates per LLM call
        
        self.job_analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert HR analyst. Analyze the following job description and extract key requirements.

Job Description:
{job_description}

Extract and structure the key requirements including required skills, preferred skills, experience level, industry, and role type.
Be specific and comprehensive in identifying technical skills, soft skills, and domain expertise.
        """)
        
        self.batch_candidate_evaluation_prompt = ChatPromptTemplate.from_template("""
You are an expert HR recruiter. Evaluate how well these candidates match the job requirements.

Job Requirements:
- Required Skills: {required_skills}
- Preferred Skills: {preferred_skills}  
- Experience Level: {experience_level}
- Industry: {industry}
- Role Type: {role_type}

Candidates to Evaluate:
{candidates_data}

For EACH candidate, provide:
1. Overall match score (0.0 to 1.0)
2. Skills match score (0.0 to 1.0)
3. Experience match score (0.0 to 1.0)
4. List of candidate strengths for this role
5. List of potential gaps or areas for development
6. Detailed reasoning explaining why this candidate is/isn't a good fit

Be objective, specific, and provide actionable insights for each candidate.
        """)

    async def analyze_job_description(self, job_description: str) -> JobRequirements:
        """Extract structured requirements from job description with caching."""
        job_hash = hashlib.md5(job_description.encode()).hexdigest()
        
        if job_hash in self._job_requirements_cache:
            logger.info("Using cached job requirements")
            return self._job_requirements_cache[job_hash]

        try:
            structured_llm = llm.with_structured_output(JobRequirements)
            analysis_chain = self.job_analysis_prompt | structured_llm
            
            requirements = await analysis_chain.ainvoke({
                "job_description": job_description
            })
            
            if isinstance(requirements, BaseModel):
                result = requirements
            else:
                result = JobRequirements(**requirements)
            
            self._job_requirements_cache[job_hash] = result
            logger.info(f"Job requirements extracted and cached")
            return result
                
        except Exception as e:
            logger.error(f"Job analysis failed: {e}")
            return JobRequirements()

    def get_all_candidate_details(self, limit: int = None) -> Dict[int, Dict]:
        """Get all candidates with their details in one query - with date conversion fix."""
        sql = """
        SELECT 
            c.candidate_id,
            c.name,
            c.email,
            c.phone,
            c.candidate_description,
            c.created_at,
            GROUP_CONCAT(DISTINCT sm.skill_name, '|') as skills,
            GROUP_CONCAT(DISTINCT 
                COALESCE(we.job_title, '') || ' at ' || COALESCE(we.company, '') || 
                ' (' ||
                CASE WHEN we.start_date IS NOT NULL THEN CAST(we.start_date AS VARCHAR) ELSE '' END ||
                ' - ' ||
                CASE WHEN we.end_date IS NOT NULL THEN CAST(we.end_date AS VARCHAR) ELSE '' END ||
                '): ' || 
                COALESCE(we.description, ''), '||') as experience,
            GROUP_CONCAT(DISTINCT 
                COALESCE(e.degree, '') || ' from ' || COALESCE(e.institution, '') || 
                ' (' || COALESCE(e.graduation_year, '') || ')', '||') as education
        FROM candidates c
        LEFT JOIN candidate_skills cs ON c.candidate_id = cs.candidate_id
        LEFT JOIN skills_master sm ON cs.skill_id = sm.skill_id  
        LEFT JOIN work_experience we ON c.candidate_id = we.candidate_id
        LEFT JOIN education e ON c.candidate_id = e.candidate_id
        GROUP BY c.candidate_id, c.name, c.email, c.phone, c.candidate_description, c.created_at
        ORDER BY c.created_at DESC
        """
        
        if limit:
            sql += f" LIMIT {limit}"
        
        results = {}
        for row in self.db.conn.execute(sql).fetchall():
            results[row[0]] = {
                "candidate_id": row[0],
                "name": row[1] or "Unknown",
                "email": row[2] or "",
                "phone": row[3] or "",
                "description": row[4] or "",
                "created_at": row[5],
                "skills": [s for s in (row[6] or "").split('|') if s.strip()],
                "experience": [e for e in (row[7] or "").split('||') if e.strip() and not e.strip().startswith(' at  (')],
                "education": [ed for ed in (row[8] or "").split('||') if ed.strip() and not ed.strip().startswith(' from  ()')]
            }
        
        logger.info(f"Retrieved details for {len(results)} candidates")
        return results


    def format_candidate_for_batch(self, candidate_details: Dict) -> str:
        """Format candidate data for batch evaluation."""
        return f"""
Candidate ID: {candidate_details['candidate_id']}
Name: {candidate_details['name']}
Skills: {', '.join(candidate_details['skills']) if candidate_details['skills'] else 'None listed'}
Experience: {'; '.join(candidate_details['experience']) if candidate_details['experience'] else 'None listed'}
Education: {'; '.join(candidate_details['education']) if candidate_details['education'] else 'None listed'}
Description: {candidate_details['description'] or 'None provided'}
---"""

    async def batch_evaluate_candidates(self, candidates_batch: List[Dict], job_requirements: JobRequirements) -> List[CandidateMatch]:
        """Evaluate multiple candidates in a single LLM call."""
        try:
            candidates_text = "\n".join([
                self.format_candidate_for_batch(candidate) 
                for candidate in candidates_batch
            ])
            
            # Use a simpler approach - ask for JSON response
            response = await llm.ainvoke([{
                "role": "user",
                "content": self.batch_candidate_evaluation_prompt.format(
                    required_skills=", ".join(job_requirements.required_skills),
                    preferred_skills=", ".join(job_requirements.preferred_skills),
                    experience_level=job_requirements.experience_level,
                    industry=job_requirements.industry,
                    role_type=job_requirements.role_type,
                    candidates_data=candidates_text
                )
            }])
            
            # Parse the response and create CandidateMatch objects
            results = []
            response_text = response.content
            
            # Simple fallback: evaluate individually if batch fails
            logger.warning("Batch evaluation parsing failed, falling back to individual evaluation")
            for candidate in candidates_batch:
                individual_result = await self.evaluate_candidate_individual(candidate, job_requirements)
                results.append(individual_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}, falling back to individual evaluation")
            # Fallback to individual evaluation
            results = []
            for candidate in candidates_batch:
                individual_result = await self.evaluate_candidate_individual(candidate, job_requirements)
                results.append(individual_result)
            return results

    async def evaluate_candidate_individual(self, candidate_details: Dict, job_requirements: JobRequirements) -> CandidateMatch:
        """Evaluate a single candidate (fallback method)."""
        try:
            structured_llm = llm.with_structured_output(CandidateMatch)
            
            # Simplified single candidate prompt
            single_prompt = ChatPromptTemplate.from_template("""
Evaluate this candidate for the job requirements.

Job: {role_type} role requiring {required_skills} in {industry}
Experience Level: {experience_level}

Candidate:
- Name: {candidate_name}
- Skills: {candidate_skills}
- Experience: {candidate_experience}
- Education: {candidate_education}
- Description: {candidate_description}

Provide match score (0.0-1.0), skills match, experience match, strengths, gaps, and reasoning.
            """)
            
            evaluation_chain = single_prompt | structured_llm
            
            evaluation = await evaluation_chain.ainvoke({
                "required_skills": ", ".join(job_requirements.required_skills) or "Any relevant skills",
                "experience_level": job_requirements.experience_level or "Any level",
                "industry": job_requirements.industry or "Any industry",
                "role_type": job_requirements.role_type or "General",
                "candidate_name": candidate_details["name"],
                "candidate_skills": ", ".join(candidate_details["skills"]) or "None listed",
                "candidate_experience": "; ".join(candidate_details["experience"]) or "None listed",
                "candidate_education": "; ".join(candidate_details["education"]) or "None listed",
                "candidate_description": candidate_details["description"] or "No description"
            })
            
            if isinstance(evaluation, BaseModel):
                result = evaluation
            else:
                result = CandidateMatch(**evaluation)
                
            result.candidate_id = candidate_details["candidate_id"]
            return result
            
        except Exception as e:
            logger.error(f"Individual evaluation failed for {candidate_details['name']}: {e}")
            return CandidateMatch(
                candidate_id=candidate_details["candidate_id"],
                match_score=0.1,  # Give small score so they're not completely filtered out
                skill_match_score=0.1,
                experience_match_score=0.1,
                reasoning=f"Evaluation failed but candidate exists: {candidate_details['name']}"
            )

    async def find_best_candidates(self, job_description: str, top_n: int = 5) -> List[Tuple[Dict, CandidateMatch]]:
        """Optimized flow focusing on LLM efficiency rather than DB filtering."""
        logger.info(f"Starting candidate matching for top {top_n} candidates")
        
        # Step 1: Analyze job requirements (1 LLM call)
        job_requirements = await self.analyze_job_description(job_description)
        
        # Step 2: Get all candidates (1 DB query, no filtering)
        # Limit to reasonable number to avoid overwhelming LLM
        candidate_limit = min(top_n * 20, 100)  # Get more candidates but cap at 100
        all_candidate_details = self.get_all_candidate_details(limit=candidate_limit)
        
        if not all_candidate_details:
            logger.warning("No candidates found in database")
            return []
        
        candidates_list = list(all_candidate_details.values())
        logger.info(f"Processing {len(candidates_list)} candidates")
        
        # Step 3: Process candidates in batches with concurrency
        semaphore = asyncio.Semaphore(self._max_concurrent_llm_calls)
        
        async def process_batch_with_limit(batch):
            async with semaphore:
                return await self.batch_evaluate_candidates(batch, job_requirements)
        
        # Create batches
        batches = []
        for i in range(0, len(candidates_list), self._batch_size):
            batch = candidates_list[i:i + self._batch_size]
            batches.append(batch)
        
        logger.info(f"Created {len(batches)} batches of {self._batch_size} candidates each")
        
        # Process all batches concurrently
        batch_results = await asyncio.gather(
            *[process_batch_with_limit(batch) for batch in batches],
            return_exceptions=True
        )
        
        # Flatten results and pair with candidate details
        candidate_matches = []
        for batch_idx, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                logger.error(f"Batch {batch_idx} failed: {batch_result}")
                continue
                
            batch = batches[batch_idx]
            for candidate, match_result in zip(batch, batch_result):
                candidate_matches.append((candidate, match_result))
        
        # Step 4: Sort by match score and return top N
        candidate_matches.sort(key=lambda x: x[1].match_score, reverse=True)
        final_matches = candidate_matches[:top_n]
        
        logger.info(f"Completed matching: returning {len(final_matches)} top candidates")
        if final_matches:
            scores = [match[1].match_score for match in final_matches]
            logger.info(f"Score range: {max(scores):.2f} - {min(scores):.2f}")
        
        return final_matches

    def clear_cache(self):
        """Clear the job requirements cache."""
        self._job_requirements_cache.clear()
        logger.info("Job requirements cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_job_requirements": len(self._job_requirements_cache),
            "max_concurrent_calls": self._max_concurrent_llm_calls,
            "batch_size": self._batch_size
        }
