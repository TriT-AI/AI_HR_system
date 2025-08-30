import streamlit as st
import utils  # Import shared utilities (CSS, theme, session, helpers)
import pandas as pd
import altair as alt
from typing import List, Tuple


# -------------------------
# Cached data access helpers
# -------------------------
@st.cache_data(ttl=60)
def fetch_top_skills(limit: int = 50) -> pd.DataFrame:
    sql = """
    SELECT sm.skill_name AS skill, COUNT(*) AS cnt
    FROM skills_master sm
    JOIN candidate_skills cs ON sm.skill_id = cs.skill_id
    GROUP BY sm.skill_name
    ORDER BY cnt DESC
    LIMIT ?
    """
    return st.session_state.db.conn.execute(sql, [limit]).fetchdf()


@st.cache_data(ttl=60)
def get_filtered_candidates(
    job_titles: List[str], min_years_exp: int, top_skills: List[str]
) -> pd.DataFrame:
    """
    Returns dataframe with columns:
      candidate_id, name, skills_matched, years_experience
    Filters:
      - job_titles: list of keywords OR-matched on work_experience.job_title
      - min_years_exp: minimum total years
      - top_skills: candidate must have at least one of these skills (IN)
    """
    conn = st.session_state.db.conn

    # Build skill subquery (parameterized)
    skill_filter_sql = ""
    skill_params: List[str] = []
    if top_skills:
        placeholders = ",".join(["?"] * len(top_skills))
        skill_filter_sql = (
            f"SELECT candidate_id FROM candidate_skills cs "
            f"JOIN skills_master sm ON cs.skill_id = sm.skill_id "
            f"WHERE LOWER(sm.skill_name) IN ({placeholders})"
        )
        skill_params = [s.lower() for s in top_skills]

    # Job title OR conditions (parameterized)
    job_title_conditions: List[str] = []
    job_title_params: List[str] = []
    if job_titles:
        for jt in job_titles:
            job_title_conditions.append("LOWER(we.job_title) LIKE ?")
            job_title_params.append(f"%{jt.lower()}%")
    job_title_sql = (
        " AND (" + " OR ".join(job_title_conditions) + ")"
        if job_title_conditions
        else ""
    )

    # Params order: job titles, skills, min years
    sql_params: List = job_title_params + skill_params + [min_years_exp]

    sql = f"""
    SELECT
        c.candidate_id,
        c.name,
        COUNT(DISTINCT sm.skill_name) AS skills_matched,
        SUM(COALESCE(JULIAN(COALESCE(we.end_date, CURRENT_DATE)) - JULIAN(we.start_date), 0)) / 365.25 AS years_experience
    FROM candidates c
    LEFT JOIN work_experience we ON c.candidate_id = we.candidate_id
    LEFT JOIN candidate_skills cs ON c.candidate_id = cs.candidate_id
    LEFT JOIN skills_master sm ON cs.skill_id = sm.skill_id
    WHERE 1=1
    {job_title_sql}
    """

    if skill_filter_sql:
        sql += f" AND c.candidate_id IN ({skill_filter_sql})"

    sql += " GROUP BY c.candidate_id, c.name"
    sql += " HAVING years_experience >= ?"
    sql += " ORDER BY skills_matched DESC, years_experience DESC"

    return conn.execute(sql, sql_params).fetchdf()


@st.cache_data(ttl=60)
def fetch_skills_for_candidates(
    candidate_ids: List[int], top_n: int = 20
) -> pd.DataFrame:
    if not candidate_ids:
        return pd.DataFrame(columns=["skill", "cnt"])
    sql = f"""
    SELECT sm.skill_name AS skill, COUNT(*) AS cnt
    FROM skills_master sm
    JOIN candidate_skills cs ON sm.skill_id = cs.skill_id
    WHERE cs.candidate_id IN ({','.join(['?'] * len(candidate_ids))})
    GROUP BY sm.skill_name
    ORDER BY cnt DESC
    LIMIT ?
    """
    params = candidate_ids + [top_n]
    return st.session_state.db.conn.execute(sql, params).fetchdf()


@st.cache_data(ttl=60)
def fetch_job_title_breakdown(
    candidate_ids: List[int], top_n: int = 12
) -> pd.DataFrame:
    if not candidate_ids:
        return pd.DataFrame(columns=["job_title", "cnt"])
    sql = f"""
    SELECT LOWER(COALESCE(we.job_title, '')) AS job_title, COUNT(*) AS cnt
    FROM work_experience we
    WHERE we.candidate_id IN ({','.join(['?'] * len(candidate_ids))})
      AND we.job_title IS NOT NULL AND we.job_title <> ''
    GROUP BY LOWER(COALESCE(we.job_title, ''))
    ORDER BY cnt DESC
    LIMIT ?
    """
    params = candidate_ids + [top_n]
    return st.session_state.db.conn.execute(sql, params).fetchdf()


@st.cache_data(ttl=60)
def fetch_inflow(candidate_ids: List[int]) -> pd.DataFrame:
    if not candidate_ids:
        return pd.DataFrame(columns=["month", "cnt"])
    sql = f"""
    SELECT strftime('%Y-%m', created_at) AS month, COUNT(*) AS cnt
    FROM candidates
    WHERE candidate_id IN ({','.join(['?'] * len(candidate_ids))})
    GROUP BY month
    ORDER BY month
    """
    return st.session_state.db.conn.execute(sql, candidate_ids).fetchdf()


# -------------------------
# UI rendering
# -------------------------
def main():
    utils.apply_theme()
    utils.init_session_state()

    stats = st.session_state.db.get_database_stats()
    utils.kpi_row(stats)

    st.header("System Dashboard")
    st.info(
        "Key insights into your candidate pool with filters for role, experience, and skills.",
        icon="💡",
    )

    # Sidebar Filters
    st.sidebar.header("Filters")
    job_titles_str = st.sidebar.text_input(
        "Job Title Keywords (comma separated)", value=""
    )
    job_titles = [jt.strip() for jt in job_titles_str.split(",") if jt.strip()]

    min_years_exp = st.sidebar.slider("Minimum Total Years Experience", 0, 40, 0)

    top_skills_df = fetch_top_skills(50)
    skills_options = top_skills_df["skill"].tolist()
    default_skills = skills_options[:5]
    selected_skills = st.sidebar.multiselect(
        "Top 5 Skills (max 5)", options=skills_options, default=default_skills
    )
    if len(selected_skills) > 5:
        st.sidebar.warning("Please select at most 5 skills.")

    apply = st.sidebar.button("Apply Filters", type="primary")

    # Active filter pills
    with st.container():
        st.markdown("### Active Filters")
        pills = []
        if job_titles:
            pills.append(
                f"<span class='tag'>Titles: {', '.join(job_titles[:3])}{'...' if len(job_titles) > 3 else ''}</span>"
            )
        pills.append(f"<span class='tag'>Min Exp: {min_years_exp}y</span>")
        if selected_skills:
            pills.append(
                f"<span class='tag'>Skills: {', '.join(selected_skills[:3])}{'...' if len(selected_skills) > 3 else ''}</span>"
            )
        if not pills:
            pills.append("<span class='tag'>No filters applied</span>")
        st.markdown(" ".join(pills), unsafe_allow_html=True)

    # Compute filtered data
    if apply or "dashboard_last_filtered" not in st.session_state:
        filtered = get_filtered_candidates(job_titles, min_years_exp, selected_skills)
        st.session_state["dashboard_last_filtered"] = filtered
    else:
        filtered = st.session_state["dashboard_last_filtered"]

    # KPIs for filtered set
    total_candidates = filtered.shape[0]
    avg_skills_matched = (
        float(filtered["skills_matched"].mean()) if total_candidates else 0.0
    )
    median_exp = (
        float(filtered["years_experience"].median()) if total_candidates else 0.0
    )
    p75_exp = (
        float(filtered["years_experience"].quantile(0.75)) if total_candidates else 0.0
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{total_candidates}</div><div class='metric-label'>Filtered Candidates</div></div>",
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{avg_skills_matched:.1f}</div><div class='metric-label'>Avg Skills Matched</div></div>",
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{median_exp:.1f}y</div><div class='metric-label'>Median Experience</div></div>",
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"<div class='metric-card'><div class='metric-value'>{p75_exp:.1f}y</div><div class='metric-label'>75th %ile Experience</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if total_candidates == 0:
        st.info("No candidates match the current filters.")
        st.markdown(
            "<div class='footer'>© 2025 HR System</div>", unsafe_allow_html=True
        )
        return

    candidate_ids = filtered["candidate_id"].tolist()

    # Layout: 2x2 grid of core charts
    c1, c2 = st.columns(2)

    # Experience distribution (histogram)
    with c1:
        st.subheader("Experience Distribution")
        exp_chart = (
            alt.Chart(filtered)
            .mark_bar()
            .encode(
                x=alt.X(
                    "years_experience:Q",
                    bin=alt.Bin(maxbins=30),
                    title="Years of Experience",
                ),
                y=alt.Y("count():Q", title="Candidates"),
                tooltip=["count()"],
            )
            .properties(height=300)
        )
        st.altair_chart(exp_chart, use_container_width=True)

    # Job title breakdown
    with c2:
        st.subheader("Top Job Titles (History)")
        job_df = fetch_job_title_breakdown(candidate_ids, 12)
        if not job_df.empty:
            job_chart = (
                alt.Chart(job_df)
                .mark_bar()
                .encode(
                    x=alt.X("cnt:Q", title="Count"),
                    y=alt.Y("job_title:N", sort="-x", title="Job Title"),
                    color=alt.Color("cnt:Q", scale=alt.Scale(scheme="blues")),
                    tooltip=["job_title", "cnt"],
                )
                .properties(height=300)
            )
            st.altair_chart(job_chart, use_container_width=True)
        else:
            st.info("No job title data for the filtered set.")

    st.markdown("---")

    # Skills distribution and scatter correlation
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Top Skills (Filtered Candidates)")
        skills_df = fetch_skills_for_candidates(candidate_ids, 20)
        if not skills_df.empty:
            skills_chart = (
                alt.Chart(skills_df)
                .mark_bar()
                .encode(
                    x=alt.X("cnt:Q", title="Count"),
                    y=alt.Y("skill:N", sort="-x", title="Skill"),
                    color=alt.Color("cnt:Q", scale=alt.Scale(scheme="greens")),
                    tooltip=["skill", "cnt"],
                )
                .properties(height=300)
            )
            st.altair_chart(skills_chart, use_container_width=True)
        else:
            st.info("No skills data for the filtered set.")

    with c4:
        st.subheader("Experience vs. Skills Matched")
        scatter = (
            alt.Chart(filtered)
            .mark_circle(size=100)
            .encode(
                x=alt.X("years_experience:Q", title="Years of Experience"),
                y=alt.Y("skills_matched:Q", title="Skills Matched"),
                tooltip=["name", "years_experience", "skills_matched"],
                color=alt.Color(
                    "years_experience:Q", scale=alt.Scale(scheme="tealblues")
                ),
            )
            .properties(height=300)
        )
        st.altair_chart(scatter, use_container_width=True)

    st.markdown("---")

    # Inflow over time (filtered)
    st.subheader("Candidate Inflow Over Time")
    inflow = fetch_inflow(candidate_ids)
    if not inflow.empty:
        inflow_chart = (
            alt.Chart(inflow)
            .mark_line(point=True)
            .encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("cnt:Q", title="Candidates"),
                tooltip=["month", "cnt"],
            )
            .interactive()
            .properties(height=280)
        )
        st.altair_chart(inflow_chart, use_container_width=True)
    else:
        st.info("No inflow data for the filtered set.")

    st.markdown("---")

    # Candidate table + export
    st.subheader("Candidate Summary")
    show_df = filtered[["name", "skills_matched", "years_experience"]].rename(
        columns={
            "name": "Candidate Name",
            "skills_matched": "Skills Matched",
            "years_experience": "Years Experience",
        }
    )
    st.dataframe(show_df, use_container_width=True)

    csv_bytes = show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV",
        data=csv_bytes,
        file_name="filtered_candidates.csv",
        mime="text/csv",
    )

    st.markdown("<div class='footer'>© 2025 HR System</div>", unsafe_allow_html=True)
