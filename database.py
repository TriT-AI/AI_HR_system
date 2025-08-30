"""
database.py
===========

• Stores résumé data in a local DuckDB file.
• Creates/updates the schema on startup (Snowflake-style star schema).
• Saves an up-to-date ER diagram as images/database_schema.png **before** the
  "Snowflake schema verified/created successfully" log line.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import duckdb
import mermaid as md          # pip install mermaid-py
from dateutil.parser import parse as dt_parse  # pip install python-dateutil

# ── logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Helpers ─────────────────────────────────────────────────────────────────────
def _to_iso_date_or_none(value) -> Optional[str]:
    """
    Convert many human date formats to ISO-8601 **YYYY-MM-DD**.
    Returns None for empty/"present" values so DB DATE columns accept NULL.
    """
    if value is None:
        return None

    s = str(value).strip()
    if not s or s.lower() in {"present", "current", "now", "ongoing", "-"}:
        return None

    # simple regex paths
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    if re.fullmatch(r"\d{4}-\d{2}", s):
        return f"{s}-01"
    if re.fullmatch(r"\d{4}", s):
        return f"{s}-01-01"

    # "Jan 2023", "March 2020" …
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", s)
    if m:
        months = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                  "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
        month = months.get(m.group(1).lower()[:3])
        if month:
            return f"{m.group(2)}-{month:02d}-01"

    # fall back to dateutil
    try:
        dt = dt_parse(s, dayfirst=False, fuzzy=True, default=datetime(1900, 1, 1))
        return dt.date().isoformat()
    except Exception:
        return None

# ── Main DB class ───────────────────────────────────────────────────────────────
class ResumeDatabase:
    def __init__(self, database_path: str = "resume_database.db"):
        self.conn = duckdb.connect(database_path, read_only=False)
        self.create_snowflake_schema()  # also writes database_schema.png

    # ── internal convenience helpers ────────────────────────────────────────
    def _exec_scalar_int(self, sql: str, params: Optional[Sequence] = None) -> int:
        self.conn.execute(sql, params or [])
        row = self.conn.fetchone()
        if row is None or row[0] is None:
            raise RuntimeError(f"Expected scalar from: {sql}")
        return int(row[0])

    def _save_schema_diagram(self, png_path: Path) -> None:
        """Render the hard-coded ER Mermaid diagram to a PNG using mermaid-py."""
        mermaid_text = '''
        erDiagram
          CANDIDATES {
            INTEGER candidate_id PK
            VARCHAR name
            VARCHAR email
            VARCHAR phone
            VARCHAR candidate_description
            TIMESTAMP created_at
          }

          WORK_EXPERIENCE {
            INTEGER work_exp_id PK
            INTEGER candidate_id FK
            VARCHAR job_title
            VARCHAR company
            DATE start_date
            DATE end_date
            TEXT description
          }

          EDUCATION {
            INTEGER education_id PK
            INTEGER candidate_id FK
            VARCHAR degree
            VARCHAR institution
            VARCHAR graduation_year
          }

          SKILLS_MASTER {
            INTEGER skill_id PK
            VARCHAR skill_name
          }

          CANDIDATE_SKILLS {
            INTEGER candidate_id FK
            INTEGER skill_id FK
          }

          PROJECTS {
            INTEGER project_id PK
            INTEGER candidate_id FK
            VARCHAR project_name
            TEXT description
          }

          CANDIDATES ||--o{ WORK_EXPERIENCE : has
          CANDIDATES ||--o{ EDUCATION : has
          CANDIDATES ||--o{ CANDIDATE_SKILLS : has
          SKILLS_MASTER ||--o{ CANDIDATE_SKILLS : has
          CANDIDATES ||--o{ PROJECTS : has
        '''
        try:
            png_path.parent.mkdir(parents=True, exist_ok=True)
            md.Mermaid(mermaid_text).to_png(str(png_path))
            logger.info("Schema diagram PNG saved to %s", png_path)
        except Exception as exc:
            logger.warning("Could not generate schema diagram: %s", exc)

    # ── DDL bootstrap (runs once per process) ───────────────────────────────
    def create_snowflake_schema(self) -> None:
        ddl = """
        CREATE SEQUENCE IF NOT EXISTS candidate_seq START 1;
        CREATE SEQUENCE IF NOT EXISTS work_exp_seq START 1;
        CREATE SEQUENCE IF NOT EXISTS education_seq START 1;
        CREATE SEQUENCE IF NOT EXISTS skill_seq START 1;
        CREATE SEQUENCE IF NOT EXISTS project_seq START 1;

        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id INTEGER PRIMARY KEY DEFAULT nextval('candidate_seq'),
            name                 VARCHAR NOT NULL,
            email                VARCHAR,
            phone                VARCHAR,
            candidate_description VARCHAR,
            created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS work_experience (
            work_exp_id INTEGER PRIMARY KEY DEFAULT nextval('work_exp_seq'),
            candidate_id INTEGER NOT NULL,
            job_title   VARCHAR,
            company     VARCHAR,
            start_date  DATE,
            end_date    DATE,
            description TEXT,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        );

        CREATE TABLE IF NOT EXISTS education (
            education_id INTEGER PRIMARY KEY DEFAULT nextval('education_seq'),
            candidate_id INTEGER NOT NULL,
            degree           VARCHAR,
            institution      VARCHAR,
            graduation_year  VARCHAR,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        );

        CREATE TABLE IF NOT EXISTS skills_master (
            skill_id   INTEGER PRIMARY KEY DEFAULT nextval('skill_seq'),
            skill_name VARCHAR UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS candidate_skills (
            candidate_id INTEGER,
            skill_id     INTEGER,
            PRIMARY KEY (candidate_id, skill_id),
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id),
            FOREIGN KEY (skill_id)     REFERENCES skills_master(skill_id)
        );

        CREATE TABLE IF NOT EXISTS projects (
            project_id INTEGER PRIMARY KEY DEFAULT nextval('project_seq'),
            candidate_id INTEGER NOT NULL,
            project_name VARCHAR,
            description  TEXT,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        );
        """
        for stmt in ddl.split(";"):
            if stmt.strip():
                self.conn.execute(stmt)

        # save diagram FIRST
        self._save_schema_diagram(Path("images/database_schema.png"))

        # finally, log success
        logger.info("Snowflake schema verified/created successfully")

    # ── ETL insert routine ──────────────────────────────────────────────────
    def import_resume_data(self, structured_resume: Dict) -> int:
        """
        Insert a parsed résumé (dict following Resume schema) into the DB.
        Returns the generated candidate_id.
        """
        try:
            self.conn.begin()

            # ---------- candidates ----------
            name = structured_resume.get("name") or ""
            contact = structured_resume.get("contact", {}) or {}
            email = contact.get("email") or ""
            phone = contact.get("phone") or ""
            candidate_description = structured_resume.get("candidate_description", "")

            self.conn.execute(
                "INSERT INTO candidates (name, email, phone, candidate_description) "
                "VALUES (?, ?, ?, ?) RETURNING candidate_id",
                [name, email, phone, candidate_description],
            )
            row = self.conn.fetchone()
            if row is None:
                raise RuntimeError("INSERT .. RETURNING gave no row")
            candidate_id = int(row[0])

            # ---------- work experience ----------
            for w in (structured_resume.get("employment", {}) or {}).get("history", []) or []:
                self.conn.execute(
                    """INSERT INTO work_experience
                       (candidate_id, job_title, company, start_date, end_date, description)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    [
                        candidate_id,
                        w.get("position") or "",
                        w.get("employer") or "",
                        _to_iso_date_or_none(w.get("start_date")),
                        _to_iso_date_or_none(w.get("end_date")),
                        w.get("summary") or "",
                    ],
                )

            # ---------- education ----------
            for edu in (structured_resume.get("education", {}) or {}).get("history", []) or []:
                self.conn.execute(
                    """INSERT INTO education
                       (candidate_id, degree, institution, graduation_year)
                       VALUES (?, ?, ?, ?)""",
                    [
                        candidate_id,
                        edu.get("degree") or "",
                        edu.get("institution") or "",
                        edu.get("graduation_year") or "",
                    ],
                )

            # ---------- skills ----------
            for skill in structured_resume.get("skills", []) or []:
                s = (skill or "").strip()
                if not s:
                    continue
                self.conn.execute(
                    "INSERT INTO skills_master (skill_name) VALUES (?) "
                    "ON CONFLICT (skill_name) DO NOTHING",
                    [s],
                )
                self.conn.execute("SELECT skill_id FROM skills_master WHERE skill_name = ?", [s])
                row = self.conn.fetchone()
                if row:
                    skill_id = int(row[0])
                    self.conn.execute(
                        "INSERT INTO candidate_skills (candidate_id, skill_id) VALUES (?, ?) "
                        "ON CONFLICT (candidate_id, skill_id) DO NOTHING",
                        [candidate_id, skill_id],
                    )

            # ---------- projects ----------
            for proj in structured_resume.get("projects", []) or []:
                self.conn.execute(
                    "INSERT INTO projects (candidate_id, project_name, description) VALUES (?, ?, ?)",
                    [candidate_id, proj.get("name") or "", proj.get("description") or ""],
                )

            self.conn.commit()
            logger.info("Imported résumé for candidate_id %s", candidate_id)
            return candidate_id

        except Exception as exc:
            self.conn.rollback()
            logger.error("Resume import failed: %s", exc)
            raise

    # ── simple read helpers ────────────────────────────────────────────────
    def search_candidates_by_skill(self, skill: str) -> List[tuple]:
        sql = """
        SELECT DISTINCT c.candidate_id,
               c.name,
               c.email,
               c.phone,
               c.candidate_description
        FROM candidates c
        JOIN candidate_skills cs ON c.candidate_id = cs.candidate_id
        JOIN skills_master  sm ON cs.skill_id     = sm.skill_id
        WHERE LOWER(sm.skill_name) LIKE LOWER(?)
        ORDER BY c.name
        """
        return self.conn.execute(sql, [f"%{skill}%"]).fetchall()

    def get_database_stats(self) -> Dict[str, int]:
        return {
            "total_candidates":        self._exec_scalar_int("SELECT COUNT(*) FROM candidates"),
            "total_work_experiences":  self._exec_scalar_int("SELECT COUNT(*) FROM work_experience"),
            "total_educations":        self._exec_scalar_int("SELECT COUNT(*) FROM education"),
            "total_skills":            self._exec_scalar_int("SELECT COUNT(*) FROM skills_master"),
            "total_projects":          self._exec_scalar_int("SELECT COUNT(*) FROM projects"),
        }

    # ── Additional methods for job matching ─────────────────────────────────
    def get_candidates_with_skills_summary(self) -> List[Dict]:
        """Get all candidates with their skills summary for matching."""
        sql = """
        SELECT 
            c.candidate_id,
            c.name,
            c.email,
            c.phone,
            c.candidate_description,
            GROUP_CONCAT(sm.skill_name, ', ') as skills
        FROM candidates c
        LEFT JOIN candidate_skills cs ON c.candidate_id = cs.candidate_id
        LEFT JOIN skills_master sm ON cs.skill_id = sm.skill_id
        GROUP BY c.candidate_id, c.name, c.email, c.phone, c.candidate_description
        ORDER BY c.created_at DESC
        """
        return [
            {
                "candidate_id": row[0],
                "name": row[1],
                "email": row[2],
                "phone": row[3],
                "description": row[4],
                "skills": row[5].split(", ") if row[5] else []
            }
            for row in self.conn.execute(sql).fetchall()
        ]

    def search_candidates_by_multiple_skills(self, skills: List[str]) -> List[tuple]:
        """Search candidates who have any of the specified skills."""
        if not skills:
            return []
        
        sql = f"""
        SELECT DISTINCT c.candidate_id,
               c.name,
               c.email,
               c.phone,
               c.candidate_description,
               COUNT(DISTINCT sm.skill_id) as matching_skills
        FROM candidates c
        JOIN candidate_skills cs ON c.candidate_id = cs.candidate_id
        JOIN skills_master sm ON cs.skill_id = sm.skill_id
        WHERE LOWER(sm.skill_name) IN ({",".join([f"LOWER(?)" for _ in skills])})
        GROUP BY c.candidate_id, c.name, c.email, c.phone, c.candidate_description
        ORDER BY matching_skills DESC, c.name
        """
        return self.conn.execute(sql, skills).fetchall()

    def close(self) -> None:
        self.conn.close()
