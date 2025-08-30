# job_matcher.py
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Tuple
from collections import Counter
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

class GapSummary(BaseModel):
    most_common_gaps: List[str] = Field(description="Most frequently mentioned gaps across candidates")
    gap_frequency: Dict[str, int] = Field(description="Gap occurrence count", default_factory=dict)
    total_candidates: int = Field(description="Total number of candidates evaluated")
    summary_insight: str = Field(description="Overall insight about candidate pool gaps")

class JobMatcher:
    def __init__(self, database):
        self.db = database
        self._job_requirements_cache = {}
        self._max_concurrent_llm_calls = 5
        self._batch_size = 2  # Reduced batch size for better parsing reliability
        
        self.job_analysis_prompt = ChatPromptTemplate.from_template("""
You are an expert HR analyst. Analyze the following job description and extract key requirements.

Job Description:
{job_description}

Extract and structure the key requirements including required skills, preferred skills, experience level, industry, and role type.
Be specific and comprehensive in identifying technical skills, soft skills, and domain expertise.
        """)
        
        self.single_candidate_evaluation_prompt = ChatPromptTemplate.from_template("""
You are an expert HR recruiter. Evaluate how well this candidate matches the job requirements.

Job Requirements:
- Required Skills: {required_skills}
- Preferred Skills: {preferred_skills}  
- Experience Level: {experience_level}
- Industry: {industry}
- Role Type: {role_type}

Candidate Profile:
- Name: {candidate_name}
- Skills: {candidate_skills}
- Experience: {candidate_experience}
- Education: {candidate_education}
- Description: {candidate_description}

Provide:
1. Overall match score (0.0 to 1.0)
2. Skills match score (0.0 to 1.0)
3. Experience match score (0.0 to 1.0)
4. List of candidate strengths for this role
5. List of potential gaps or areas for development
6. Detailed reasoning explaining why this candidate is/isn't a good fit

Be objective, specific, and provide actionable insights.
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
            logger.info("Job requirements extracted and cached")
            return result
                
        except Exception as e:
            logger.error(f"Job analysis failed: {e}")
            return JobRequirements()

    def get_all_candidate_details(self, limit: int = None) -> Dict[int, Dict]:
        """Get all candidates with their details in one query."""
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

    async def evaluate_candidate_individual(self, candidate_details: Dict, job_requirements: JobRequirements) -> CandidateMatch:
        """Evaluate a single candidate."""
        try:
            structured_llm = llm.with_structured_output(CandidateMatch)
            evaluation_chain = self.single_candidate_evaluation_prompt | structured_llm
            
            evaluation = await evaluation_chain.ainvoke({
                "required_skills": ", ".join(job_requirements.required_skills) or "Any relevant skills",
                "preferred_skills": ", ".join(job_requirements.preferred_skills) or "None specified",
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
                match_score=0.1,
                skill_match_score=0.1,
                experience_match_score=0.1,
                reasoning=f"Evaluation failed: {str(e)}",
                gaps=["Evaluation error - manual review needed"]
            )

    async def find_best_candidates(self, job_description: str, top_n: int = 5) -> List[Tuple[Dict, CandidateMatch]]:
        """Find and rank the best candidates for a job."""
        logger.info(f"Starting candidate matching for top {top_n} candidates")
        
        # Step 1: Analyze job requirements
        job_requirements = await self.analyze_job_description(job_description)
        
        # Step 2: Get all candidates
        candidate_limit = min(top_n * 20, 100)
        all_candidate_details = self.get_all_candidate_details(limit=candidate_limit)
        
        if not all_candidate_details:
            logger.warning("No candidates found in database")
            return []
        
        candidates_list = list(all_candidate_details.values())
        logger.info(f"Processing {len(candidates_list)} candidates")
        
        # Step 3: Evaluate candidates individually (more reliable than batch)
        semaphore = asyncio.Semaphore(self._max_concurrent_llm_calls)
        
        async def evaluate_with_limit(candidate):
            async with semaphore:
                return await self.evaluate_candidate_individual(candidate, job_requirements)
        
        # Process all candidates concurrently
        evaluation_tasks = [evaluate_with_limit(candidate) for candidate in candidates_list]
        evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        # Pair candidates with their evaluations
        candidate_matches = []
        for candidate, result in zip(candidates_list, evaluation_results):
            if isinstance(result, Exception):
                logger.error(f"Error evaluating candidate {candidate['name']}: {result}")
                continue
            candidate_matches.append((candidate, result))
        
        # Step 4: Sort by match score and return top N
        candidate_matches.sort(key=lambda x: x[1].match_score, reverse=True)
        final_matches = candidate_matches[:top_n]
        
        logger.info(f"Completed matching: returning {len(final_matches)} top candidates")
        if final_matches:
            scores = [match[1].match_score for match in final_matches]
            logger.info(f"Score range: {max(scores):.2f} - {min(scores):.2f}")
        
        return final_matches

    def analyze_candidate_gaps(self, candidate_matches: List[Tuple[Dict, CandidateMatch]]) -> GapSummary:
        """Analyze and summarize gaps across all matched candidates."""
        if not candidate_matches:
            return GapSummary(
                most_common_gaps=[],
                gap_frequency={},
                total_candidates=0,
                summary_insight="No candidates to analyze"
            )
        
        # Collect all gaps
        all_gaps = []
        for _, match in candidate_matches:
            all_gaps.extend(match.gaps)
        
        # Count gap frequency
        gap_counter = Counter(all_gaps)
        gap_frequency = dict(gap_counter)
        
        # Get most common gaps
        most_common = gap_counter.most_common(10)  # Top 10 most common gaps
        most_common_gaps = [gap for gap, count in most_common]
        
        # Generate summary insight
        total_candidates = len(candidate_matches)
        if most_common:
            top_gap = most_common[0][0]
            top_gap_percentage = (most_common[0][1] / total_candidates) * 100
            summary_insight = f"The most common gap is '{top_gap}' affecting {most_common[0][1]} out of {total_candidates} candidates ({top_gap_percentage:.1f}%). This suggests a talent shortage in this area."
        else:
            summary_insight = "All candidates appear well-matched with minimal gaps identified."
        
        return GapSummary(
            most_common_gaps=most_common_gaps,
            gap_frequency=gap_frequency,
            total_candidates=total_candidates,
            summary_insight=summary_insight
        )

    def get_gap_insights(self, candidate_matches: List[Tuple[Dict, CandidateMatch]]) -> List[str]:
        """Get formatted gap insights for display."""
        gap_summary = self.analyze_candidate_gaps(candidate_matches)
        
        if not gap_summary.most_common_gaps:
            return ["No significant gaps identified across candidates."]
        
        insights = [gap_summary.summary_insight]
        insights.append("")  # Empty line for spacing
        insights.append("**Most Common Gaps:**")
        
        for gap in gap_summary.most_common_gaps:
            count = gap_summary.gap_frequency[gap]
            percentage = (count / gap_summary.total_candidates) * 100
            insights.append(f"• {gap} ({count}/{gap_summary.total_candidates} candidates, {percentage:.1f}%)")
        
        return insights

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
