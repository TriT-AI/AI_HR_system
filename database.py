# database.py
import duckdb
import json
from typing import Dict, List, Optional
from pathlib import Path
import logging

import re
from datetime import datetime
from dateutil.parser import parse as dt_parse  # pip install python-dateutil


def _to_iso_date_or_none(value) -> Optional[str]:
    """
    Normalize various date strings to ISO YYYY-MM-DD or return None.
    Treats 'Present'/'Current' as None.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    low = s.lower()
    if low in {"present", "current", "now", "ongoing", "-"}:
        return None

    # Fast-path simple known shapes
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):  # YYYY-MM-DD
        return s
    if re.fullmatch(r"\d{4}-\d{2}", s):       # YYYY-MM
        return f"{s}-01"
    if re.fullmatch(r"\d{4}", s):             # YYYY
        return f"{s}-01-01"

    # Month YYYY (e.g., "Jan 2023", "March 2020")
    m = re.fullmatch(r"([A-Za-z]+)\s+(\d{4})", s)
    if m:
        mon = m.group(1).lower()[:3]
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        if mon in months:
            return f"{m.group(2)}-{months[mon]:02d}-01"

    # Generic parse using dateutil (handles "Jan 2023", "March 5, 2020", etc.)
    try:
        dt = dt_parse(s, dayfirst=False, fuzzy=True, default=datetime(1900, 1, 1))
        return dt.date().isoformat()
    except Exception:
        # Unparseable: return None instead of breaking the insert
        return None


class ResumeDatabase:
    def __init__(self, database_path: str = "resume_database.db"):
        self.conn = duckdb.connect(database=database_path, read_only=False)
        self.create_snowflake_schema()
    
    def create_snowflake_schema(self):
        schema_ddl = """
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id INTEGER PRIMARY KEY,
            name VARCHAR NOT NULL,
            email VARCHAR,
            phone VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS work_experience (
            work_exp_id INTEGER PRIMARY KEY,
            candidate_id INTEGER NOT NULL,
            job_title VARCHAR,
            company VARCHAR,
            start_date DATE,
            end_date DATE,
            description TEXT,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        );

        CREATE TABLE IF NOT EXISTS education (
            education_id INTEGER PRIMARY KEY,
            candidate_id INTEGER NOT NULL,
            degree VARCHAR,
            institution VARCHAR,
            graduation_year VARCHAR,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        );

        CREATE TABLE IF NOT EXISTS skills_master (
            skill_id INTEGER PRIMARY KEY,
            skill_name VARCHAR UNIQUE NOT NULL
        );

        CREATE TABLE IF NOT EXISTS candidate_skills (
            candidate_id INTEGER,
            skill_id INTEGER,
            PRIMARY KEY (candidate_id, skill_id),
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id),
            FOREIGN KEY (skill_id) REFERENCES skills_master(skill_id)
        );

        CREATE TABLE IF NOT EXISTS projects (
            project_id INTEGER PRIMARY KEY,
            candidate_id INTEGER NOT NULL,
            project_name VARCHAR,
            description TEXT,
            FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
        );

        CREATE SEQUENCE IF NOT EXISTS candidate_seq START 1;
        CREATE SEQUENCE IF NOT EXISTS work_exp_seq START 1;
        CREATE SEQUENCE IF NOT EXISTS education_seq START 1;
        CREATE SEQUENCE IF NOT EXISTS skill_seq START 1;
        CREATE SEQUENCE IF NOT EXISTS project_seq START 1;
        """
        for statement in schema_ddl.split(';'):
            if statement.strip():
                self.conn.execute(statement)
        logging.info("Snowflake schema verified/created successfully")

    def import_resume_data(self, structured_resume: Dict) -> int:
        try:
            self.conn.begin()

            # 1) Candidate
            candidate_id = self.conn.execute("SELECT nextval('candidate_seq')").fetchone()[0]
            name = structured_resume.get('name') or ''
            contact = structured_resume.get('contact', {}) or {}
            email = contact.get('email') or ''
            phone = contact.get('phone') or ''

            self.conn.execute(
                "INSERT INTO candidates (candidate_id, name, email, phone) VALUES (?, ?, ?, ?)",
                [candidate_id, name, email, phone]
            )

            # 2) Work experience (normalize dates)
            for work in structured_resume.get('employment', {}).get('history', []):
                job_title = work.get('position') or ''
                company = work.get('employer') or ''
                start_iso = _to_iso_date_or_none(work.get('start_date'))
                end_iso = _to_iso_date_or_none(work.get('end_date'))  # 'Present' -> None
                description = work.get('summary') or ''

                self.conn.execute(
                    """INSERT INTO work_experience
                       (work_exp_id, candidate_id, job_title, company, start_date, end_date, description)
                       VALUES (nextval('work_exp_seq'), ?, ?, ?, ?, ?, ?)""",
                    [candidate_id, job_title, company, start_iso, end_iso, description]
                )

            # 3) Education
            for edu in structured_resume.get('education', {}).get('history', []):
                degree = edu.get('degree') or ''
                institution = edu.get('institution') or ''
                graduation_year = edu.get('graduation_year') or ''

                self.conn.execute(
                    """INSERT INTO education
                       (education_id, candidate_id, degree, institution, graduation_year)
                       VALUES (nextval('education_seq'), ?, ?, ?, ?)""",
                    [candidate_id, degree, institution, graduation_year]
                )

            # 4) Skills (ensure master + link)
            for skill_name in structured_resume.get('skills', []):
                sname = (skill_name or '').strip()
                if not sname:
                    continue

                self.conn.execute(
                    "INSERT INTO skills_master (skill_id, skill_name) VALUES (nextval('skill_seq'), ?) "
                    "ON CONFLICT (skill_name) DO NOTHING",
                    [sname]
                )
                skill_id = self.conn.execute(
                    "SELECT skill_id FROM skills_master WHERE skill_name = ?", [sname]
                ).fetchone()[0]

                self.conn.execute(
                    "INSERT OR IGNORE INTO candidate_skills (candidate_id, skill_id) VALUES (?, ?)",
                    [candidate_id, skill_id]
                )

            # 5) Projects
            for project in structured_resume.get('projects', []):
                project_name = project.get('name') or ''
                description = project.get('description') or ''

                self.conn.execute(
                    """INSERT INTO projects (project_id, candidate_id, project_name, description)
                       VALUES (nextval('project_seq'), ?, ?, ?)""",
                    [candidate_id, project_name, description]
                )

            self.conn.commit()
            logging.info(f"Successfully imported resume for candidate_id: {candidate_id}")
            return candidate_id

        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error importing resume data: {e}")
            raise

    def search_candidates_by_skill(self, skill_name: str) -> List:
        query = """
        SELECT DISTINCT c.candidate_id, c.name, c.email, c.phone
        FROM candidates c
        JOIN candidate_skills cs ON c.candidate_id = cs.candidate_id
        JOIN skills_master sm ON cs.skill_id = sm.skill_id
        WHERE LOWER(sm.skill_name) LIKE LOWER(?)
        """
        return self.conn.execute(query, [f"%{skill_name}%"]).fetchall()

    def get_database_stats(self) -> Dict:
        return {
            'total_candidates': self.conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0],
            'total_work_experiences': self.conn.execute("SELECT COUNT(*) FROM work_experience").fetchone()[0],
            'total_educations': self.conn.execute("SELECT COUNT(*) FROM education").fetchone()[0],
            'total_skills': self.conn.execute("SELECT COUNT(*) FROM skills_master").fetchone()[0],
            'total_projects': self.conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        }
    
    def close(self):
        self.conn.close()
