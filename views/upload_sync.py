# upload_sync.py
import streamlit as st
import utils  # Import shared utilities


def main():
    # Apply shared theme and init session (called per page for safety)
    utils.apply_theme()
    utils.init_session_state()

    # Display shared KPIs
    stats = st.session_state.db.get_database_stats()
    utils.kpi_row(stats)

    tab1, tab2 = st.tabs(["📤 Upload CVs", "☁️ Drive Sync"])

    with tab1:
        st.header("Upload CVs")
        uploaded_files = st.file_uploader(
            "Drop files here",
            type=["pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Supports batch processing of resumes",
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} files uploaded. Ready to process?")
            if st.button("Process Now", type="primary"):
                results = []
                with st.spinner("Processing files..."):
                    for idx, file in enumerate(uploaded_files, 1):
                        utils.display_progress(
                            idx, len(uploaded_files), f"Processing: {file.name}"
                        )
                        path, bytes_data = utils.save_upload_and_cache_bytes(file)
                        try:
                            structured = utils.run_async(
                                st.session_state.processor.process_resume(str(path))
                            )
                            cand_id = st.session_state.db.import_resume_data(structured)
                            results.append(
                                {
                                    "name": file.name,
                                    "bytes": bytes_data,
                                    "candidate_id": cand_id,
                                    "data": structured,
                                    "is_pdf": file.name.lower().endswith(".pdf"),
                                }
                            )
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {e}")
                        finally:
                            path.unlink(missing_ok=True)
                st.success("Processing complete!")
                st.subheader("Results")
                for res in results:
                    with st.expander(f"{res['name']} (ID: {res['candidate_id']})"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Preview")
                            if res["is_pdf"]:
                                utils.embed_pdf_from_bytes(res["bytes"])
                            else:
                                img = utils.Image.open(utils.BytesIO(res["bytes"]))
                                st.image(img, use_column_width=True)
                        with col2:
                            st.subheader("Extracted Data")
                            st.json(res["data"])
                            utils.export_json(
                                res["data"],
                                "⬇️ Export JSON",
                                f"{res['candidate_id']}.json",
                            )
        else:
            st.warning("No files uploaded yet.")

    with tab2:
        st.header("Google Drive Sync")
        stats = st.session_state.drive_processor.get_drive_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Processed", stats["total_processed_files"])
        col2.metric("Last 7 Days", stats["files_processed_last_week"])
        col3.metric("Status", "Active" if stats["configured"] else "Inactive")
        if not stats["configured"]:
            st.error("Please configure Google Drive in settings.")
        else:
            if st.button("Sync Now"):
                with st.spinner("Syncing..."):
                    new_files = utils.run_async(
                        st.session_state.drive_processor.process_new_files()
                    )
                    if new_files:
                        st.success(f"Processed {len(new_files)} new files")
                        st.subheader(
                            "Results"
                        )  # New: Add subheader for consistency with tab1

                    for res in new_files:
                        with st.expander(
                            f"{res['file_name']} (ID: {res['candidate_id']})"
                        ):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("Preview")
                                if "bytes" not in res:
                                    st.warning(
                                        "Preview unavailable (bytes data missing)."
                                    )
                                    continue  # Skip preview if bytes are missing
                                # Determine if PDF (use key if present, else fallback to filename)
                                is_pdf = res.get(
                                    "is_pdf", res["file_name"].lower().endswith(".pdf")
                                )
                                if is_pdf:
                                    utils.embed_pdf_from_bytes(res["bytes"])
                                else:
                                    img = utils.Image.open(
                                        utils.BytesIO(res["bytes"])
                                    )  # Assumes utils has Image and BytesIO
                                    st.image(img, use_column_width=True)
                            with col2:
                                st.subheader("Extracted Data")
                                st.json(res["data"])
                                utils.export_json(
                                    res["data"],
                                    "⬇️ Export JSON",
                                    f"{res['candidate_id']}.json",
                                )

                    else:
                        st.info("No new files.")

    # Footer
    st.markdown("<div class='footer'>© 2025 HR System</div>", unsafe_allow_html=True)
