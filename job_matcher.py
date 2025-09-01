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

# Initialize LLM (unchanged)
llm = AzureChatOpenAI(
    api_key=settings.AZURE_OPENAI_API_KEY,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    azure_deployment=settings.AZURE_OPENAI_DEPLOYMENT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)


# Models (unchanged)
class JobRequirements(BaseModel):
    required_skills: List[str] = Field(
        description="Essential skills for the job", default_factory=list
    )
    preferred_skills: List[str] = Field(
        description="Nice-to-have skills", default_factory=list
    )
    experience_level: str = Field(description="Required experience level", default="")
    industry: str = Field(description="Industry or domain", default="")
    role_type: str = Field(
        description="Type of role (technical, managerial, etc.)", default=""
    )


class CandidateMatch(BaseModel):
    candidate_id: int = Field(description="Database ID of the candidate")
    match_score: float = Field(
        description="Match score from 0.0 to 1.0", ge=0.0, le=1.0
    )
    skill_match_score: float = Field(
        description="Skills compatibility score", ge=0.0, le=1.0
    )
    experience_match_score: float = Field(
        description="Experience level match", ge=0.0, le=1.0
    )
    strengths: List[str] = Field(
        description="Candidate's strengths for this role", default_factory=list
    )
    gaps: List[str] = Field(
        description="Areas where candidate may need development", default_factory=list
    )
    reason: str = Field(
        description="Short explanation of why this candidate is a good fit and why not",
        default="",
    )


class GapSummary(BaseModel):
    most_common_gaps: Optional[List[str]] = Field(
        description="Most frequently mentioned gaps across candidates", default=None
    )
    gap_frequency: Optional[Dict[str, int]] = Field(
        description="Gap occurrence count", default=None
    )
    total_candidates: Optional[int] = Field(
        description="Total number of candidates evaluated", default=None
    )
    summary_insight: Optional[str] = Field(
        description="Overall insight about candidate pool gaps, highlight key areas for improvement",
        default=None,
    )


class JobMatcher:
    def __init__(self, database):
        self.db = database
        self._job_requirements_cache = {}
        self._max_concurrent_llm_calls = 5
        self._batch_size = 2  # Reduced batch size for better parsing reliability

        # Prompts (unchanged)
        self.job_analysis_prompt = ChatPromptTemplate.from_template(
            """
You are an expert HR analyst. Analyze the following job description and extract key requirements.

Job Description:
{job_description}

Extract and structure the key requirements including required skills, preferred skills, experience level, industry, and role type.
Be specific and comprehensive in identifying technical skills, soft skills, and domain expertise.
        """
        )

        self.single_candidate_evaluation_prompt = ChatPromptTemplate.from_template(
            """
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
6. Summary reason explaining why this candidate is/isn't a good fit

Be objective, specific, and provide actionable insights.
        """
        )

        # NEW: Prompt for AI-powered gap summary on shortlist
        self.gap_summary_prompt = ChatPromptTemplate.from_template(
            """
You are an expert HR analyst. Analyze the gaps across this shortlist of candidates for the job.

Job Description: {job_description}

Shortlisted Candidates and Their Gaps:
{candidate_gaps}

Provide a structured summary INCLUDING ALL THESE FIELDS, even if empty:
- most_common_gaps: List of the most common gaps (empty list if none)
- gap_frequency: Dictionary of gap counts (empty dict if none)
- total_candidates: Number of candidates analyzed
- summary_insight: Overall narrative insight (e.g., "No major gaps" if none)
"""
        )

    # analyze_job_description (unchanged)
    async def analyze_job_description(self, job_description: str) -> JobRequirements:
        job_hash = hashlib.md5(job_description.encode()).hexdigest()

        if job_hash in self._job_requirements_cache:
            logger.info("Using cached job requirements")
            return self._job_requirements_cache[job_hash]

        try:
            structured_llm = llm.with_structured_output(JobRequirements)
            analysis_chain = self.job_analysis_prompt | structured_llm

            requirements = await analysis_chain.ainvoke(
                {"job_description": job_description}
            )

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

    # get_all_candidate_details (enhanced: configurable limit, default to all)
    def get_all_candidate_details(self, limit: Optional[int] = None) -> Dict[int, Dict]:
        """Get all candidates with their details in one query. Limit optional."""
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
                "skills": [s for s in (row[6] or "").split("|") if s.strip()],
                "experience": [
                    e
                    for e in (row[7] or "").split("||")
                    if e.strip() and not e.strip().startswith(" at  (")
                ],
                "education": [
                    ed
                    for ed in (row[8] or "").split("||")
                    if ed.strip() and not ed.strip().startswith(" from  ()")
                ],
            }

        logger.info(f"Retrieved details for {len(results)} candidates")
        return results

    # evaluate_candidate_individual (unchanged)
    async def evaluate_candidate_individual(
        self, candidate_details: Dict, job_requirements: JobRequirements
    ) -> CandidateMatch:
        try:
            structured_llm = llm.with_structured_output(CandidateMatch)
            evaluation_chain = self.single_candidate_evaluation_prompt | structured_llm

            evaluation = await evaluation_chain.ainvoke(
                {
                    "required_skills": ", ".join(job_requirements.required_skills)
                    or "Any relevant skills",
                    "preferred_skills": ", ".join(job_requirements.preferred_skills)
                    or "None specified",
                    "experience_level": job_requirements.experience_level
                    or "Any level",
                    "industry": job_requirements.industry or "Any industry",
                    "role_type": job_requirements.role_type or "General",
                    "candidate_name": candidate_details["name"],
                    "candidate_skills": ", ".join(candidate_details["skills"])
                    or "None listed",
                    "candidate_experience": "; ".join(candidate_details["experience"])
                    or "None listed",
                    "candidate_education": "; ".join(candidate_details["education"])
                    or "None listed",
                    "candidate_description": candidate_details["description"]
                    or "No description",
                }
            )

            if isinstance(evaluation, BaseModel):
                result = evaluation
            else:
                result = CandidateMatch(**evaluation)

            result.candidate_id = candidate_details["candidate_id"]
            return result

        except Exception as e:
            logger.error(
                f"Individual evaluation failed for {candidate_details['name']}: {e}"
            )
            return CandidateMatch(
                candidate_id=candidate_details["candidate_id"],
                match_score=0.1,
                skill_match_score=0.1,
                experience_match_score=0.1,
                reason=f"Evaluation failed: {str(e)}",
                gaps=["Evaluation error - manual review needed"],
            )

    # Enhanced: find_best_candidates now supports min_score, fetches more candidates, filters internally
    async def find_best_candidates(
        self,
        job_description: str,
        top_n: int = 5,
        min_score: float = 0.0,
        candidate_limit: Optional[int] = None,
    ) -> List[Tuple[Dict, CandidateMatch]]:
        """Find and rank the best candidates for a job. Scores all (or limited), filters by min_score, returns top N."""
        logger.info(
            f"Starting candidate matching for top {top_n} with min_score {min_score}"
        )

        # Step 1: Analyze job requirements
        job_requirements = await self.analyze_job_description(job_description)

        # Step 2: Get candidates (all or limited)
        all_candidate_details = self.get_all_candidate_details(limit=candidate_limit)

        if not all_candidate_details:
            logger.warning("No candidates found in database")
            return []

        candidates_list = list(all_candidate_details.values())
        logger.info(f"Processing {len(candidates_list)} candidates")

        # Step 3: Evaluate all candidates concurrently for scoring
        semaphore = asyncio.Semaphore(self._max_concurrent_llm_calls)

        async def evaluate_with_limit(candidate):
            async with semaphore:
                return await self.evaluate_candidate_individual(
                    candidate, job_requirements
                )

        evaluation_tasks = [
            evaluate_with_limit(candidate) for candidate in candidates_list
        ]
        evaluation_results = await asyncio.gather(
            *evaluation_tasks, return_exceptions=True
        )

        # Pair and filter by min_score
        candidate_matches = []
        for candidate, result in zip(candidates_list, evaluation_results):
            if isinstance(result, Exception):
                logger.error(
                    f"Error evaluating candidate {candidate['name']}: {result}"
                )
                continue
            if result.match_score >= min_score:
                candidate_matches.append((candidate, result))

        # Step 4: Sort by match_score and take top N
        candidate_matches.sort(key=lambda x: x[1].match_score, reverse=True)
        final_matches = candidate_matches[:top_n]

        logger.info(
            f"Completed matching: {len(candidate_matches)} above min_score, returning top {len(final_matches)}"
        )
        if final_matches:
            scores = [match[1].match_score for match in final_matches]
            logger.info(f"Score range: {max(scores):.2f} - {min(scores):.2f}")

        return final_matches

    # NEW: AI-powered gap summary on shortlist
    async def analyze_candidate_gaps_ai(
        self, candidate_matches: List[Tuple[Dict, CandidateMatch]], job_description: str
    ) -> GapSummary:
        """Use AI to generate insightful gap summary for the shortlist."""
        if not candidate_matches:
            return GapSummary(
                most_common_gaps=[],
                gap_frequency={},
                total_candidates=0,
                summary_insight="No candidates to analyze",
            )

        # Prepare input: Aggregate gaps from shortlist
        candidate_gaps = "\n".join(
            [
                f"Candidate {cand['name']}: Gaps - {', '.join(match.gaps)}"
                for cand, match in candidate_matches
            ]
        )

        try:
            structured_llm = llm.with_structured_output(GapSummary)
            summary_chain = self.gap_summary_prompt | structured_llm

            summary = await summary_chain.ainvoke(
                {
                    "job_description": job_description,
                    "candidate_gaps": candidate_gaps or "No gaps identified",
                }
            )

            if isinstance(summary, BaseModel):
                result = summary
            else:
                result = GapSummary(**summary)

            result.total_candidates = len(candidate_matches)
            return result

        except Exception as e:
            logger.error(f"AI gap summary failed: {e}")
            return GapSummary(
                total_candidates=len(candidate_matches),
                summary_insight=f"Analysis failed: {str(e)}",
            )

    # Enhanced: Now uses AI summary if flag set, else fallback to programmatic
    async def analyze_candidate_gaps(
        self,
        candidate_matches: List[Tuple[Dict, CandidateMatch]],
        job_description: str = "",
        use_ai: bool = True,
    ) -> GapSummary:
        """Analyze and summarize gaps. Use AI for shortlist if specified."""
        if use_ai and job_description:
            return await self.analyze_candidate_gaps_ai(
                candidate_matches, job_description
            )

        # Fallback programmatic aggregation (original logic)
        if not candidate_matches:
            return GapSummary(
                most_common_gaps=[],
                gap_frequency={},
                total_candidates=0,
                summary_insight="No candidates to analyze",
            )

        all_gaps = []
        for _, match in candidate_matches:
            all_gaps.extend(match.gaps)

        gap_counter = Counter(all_gaps)
        gap_frequency = dict(gap_counter)

        most_common = gap_counter.most_common(10)
        most_common_gaps = [gap for gap, count in most_common]

        total_candidates = len(candidate_matches)
        if most_common:
            top_gap = most_common[0][0]
            top_gap_percentage = (most_common[0][1] / total_candidates) * 100
            summary_insight = f"The most common gap is '{top_gap}' affecting {most_common[0][1]} out of {total_candidates} candidates ({top_gap_percentage:.1f}%). This suggests a talent shortage in this area."
        else:
            summary_insight = (
                "All candidates appear well-matched with minimal gaps identified."
            )

        return GapSummary(
            most_common_gaps=most_common_gaps,
            gap_frequency=gap_frequency,
            total_candidates=total_candidates,
            summary_insight=summary_insight,
        )

    # get_gap_insights (updated to use new analyze method)
    async def get_gap_insights(
        self,
        candidate_matches: List[Tuple[Dict, CandidateMatch]],
        job_description: str = "",
    ) -> List[str]:
        """Get formatted gap insights for display, using AI if possible."""
        gap_summary = await self.analyze_candidate_gaps(
            candidate_matches, job_description
        )

        if not gap_summary.most_common_gaps:
            return ["No significant gaps identified across candidates."]

        insights = [gap_summary.summary_insight]
        insights.append("")  # Empty line for spacing
        insights.append("**Most Common Gaps:**")

        for gap in gap_summary.most_common_gaps:
            count = gap_summary.gap_frequency[gap]
            percentage = (count / gap_summary.total_candidates) * 100
            insights.append(
                f"• {gap} ({count}/{gap_summary.total_candidates} candidates, {percentage:.1f}%)"
            )

        return insights

    # clear_cache and get_cache_stats (unchanged)
    def clear_cache(self):
        self._job_requirements_cache.clear()
        logger.info("Job requirements cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        return {
            "cached_job_requirements": len(self._job_requirements_cache),
            "max_concurrent_calls": self._max_concurrent_llm_calls,
            "batch_size": self._batch_size,
        }
