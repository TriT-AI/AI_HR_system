import streamlit as st
import utils  # Import shared utilities


def main():
    utils.apply_theme()
    utils.init_session_state()

    stats = st.session_state.db.get_database_stats()
    utils.kpi_row(stats)

    tab1, tab2 = st.tabs(["🔎 Search Candidates", "🤖 Job Matching"])

    with tab1:
        st.header("Search Candidates")
        st.info(
            "Search by skills, experience, or other criteria. Results are ranked by relevance.",
            icon="💡",
        )
        col1, col2 = st.columns(2)
        with col1:
            skill = st.text_input(
                "Skill Keyword",
                placeholder="e.g., Python, SQL",
                help="Search for specific skills",
            )
        with col2:
            min_experience = st.number_input(
                "Min Years Experience", min_value=0, value=0
            )

        if st.button("Search", type="primary"):
            with st.spinner("Searching..."):
                rows = st.session_state.db.search_candidates_by_skill_and_experience(
                    skill, min_experience
                )  # Enhanced method

            if rows:
                st.success(f"Found {len(rows)} candidates")
                st.subheader("Candidate List")
                # List candidates with clickable detail expanders
                for cid, name, email, phone, desc in rows:
                    with st.expander(f"{name} (ID: {cid})"):
                        st.markdown(
                            f"""
                            <div class="card">
                                <p><b>Summary:</b> {desc or 'No summary available'}</p>
                                <p>Email: {email} | Phone: {phone}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            else:
                st.info("No candidates found. Try broadening your search.")

    with tab2:
        st.header("AI Job Matching")
        st.info(
            "Enter job details for AI-powered matching. Highlights include match scores and gap analysis.",
            icon="💡",
        )
        job_desc = st.text_area(
            "Job Description",
            height=200,
            placeholder="Describe the role, required skills, experience...",
        )
        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Number of top candidates", 1, 20, 5)
        with col2:
            min_score = st.slider("Minimum Match Score", 0.0, 1.0, 0.5)
        if st.button("Match Candidates", type="primary"):
            with st.spinner("Analyzing and matching..."):
                try:
                    # Use run_async for the async find_best_candidates
                    matches = utils.run_async(
                        st.session_state.job_matcher.find_best_candidates(
                            job_desc, top_n, min_score
                        )
                    )
                    if matches:
                        st.success(
                            f"Found {len(matches)} matching candidates above {min_score*100:.0f}% score"
                        )
                        st.markdown("---")
                        st.subheader("📊 Candidate Pool Analysis")

                        # Use run_async for the async get_gap_insights
                        gap_insights = utils.run_async(
                            st.session_state.job_matcher.get_gap_insights(
                                matches, job_desc
                            )
                        )
                        for insight in gap_insights:
                            if insight.startswith("**"):
                                st.markdown(insight)
                            elif insight.strip() == "":
                                st.write("")
                            else:
                                st.write(insight)

                        with st.expander("🔍 Detailed Gap Statistics"):
                            # If analyze_candidate_gaps is called separately, wrap it too if needed
                            gap_summary = utils.run_async(
                                st.session_state.job_matcher.analyze_candidate_gaps(
                                    matches, job_desc
                                )
                            )
                            st.json(gap_summary.model_dump())

                        st.subheader("Candidate Matches")
                        # Show high-level list with expanders for details
                        for i, (cand, match) in enumerate(matches, 1):
                            with st.expander(
                                f"#{i} {cand['name']} - Score: {match.match_score:.0%}"
                            ):
                                st.markdown(
                                    f"""
                                    <div class="card">
                                        <p class="muted">Email: {cand['email']} | Phone: {cand['phone']}</p>
                                        <div>Skills: {', '.join(cand['skills'][:5]) + ('...' if len(cand['skills']) > 5 else '')}</div>
                                        <div class="progress-bar"><div class="progress-fill" style="width:{match.match_score * 100}%"></div></div>
                                        <h5>Reason</h5>
                                        <p>{match.reason}</p>
                                        <h5>Strengths</h5>
                                        {' '.join([f'<span class="tag">{s}</span>' for s in match.strengths])}
                                        <h5>Gaps</h5>
                                        {' '.join([f'<span class="warning-tag">{g}</span>' for g in match.gaps])}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.warning("No candidates meet the criteria.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("<div class='footer'>© 2025 HR System</div>", unsafe_allow_html=True)
