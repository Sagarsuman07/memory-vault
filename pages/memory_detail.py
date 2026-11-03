import html
import mimetypes

import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Memory Detail",
    page_icon="🧠",
    layout="wide",
)

from config.settings import settings
from security.access_control import require_private_access
from database.database import get_connection, initialize_database

from memory.memory_service import (
    get_memory_by_id,
    generate_memory_summary,
    update_memory,
    delete_memory,
)

from storage.file_store import generate_signed_url



require_private_access()

initialize_database()

USER_ID = settings.DEMO_USER_ID


# ============================================================
# Layout constants
# ============================================================

# Total visible height of BOTH boxes (Original Content / Summary).
# Using st.container(height=...) makes this a true fixed height —
# unlike plain CSS, it won't shrink for short content.
BOX_HEIGHT = 460

# Height of the inner scrollable area inside the Summary box.
# Kept smaller than BOX_HEIGHT to leave room for the fixed
# "Summary" heading above it, so only the value scrolls.
SUMMARY_INNER_HEIGHT = 340


# ============================================================
# Minimal Targeted Styling
# ============================================================

st.markdown(
    """
    <style>
        /* Cap preview sizes so large images/docs don't blow up the layout */
        [data-testid="stImage"] img {
            max-height: 380px;
            width: auto;
            object-fit: contain;
            border-radius: 8px;
        }

        .doc-preview-link {
            display: block;
            text-decoration: none;
            color: inherit;
        }

        .doc-preview-box {
            border: 1px solid rgba(250, 250, 250, 0.15);
            border-radius: 8px;
            padding: 1.25rem;
            text-align: center;
            transition: background-color 0.15s ease-in-out;
        }

        .doc-preview-box:hover {
            background-color: rgba(250, 250, 250, 0.05);
        }

        /* Red Delete button — targeted via marker + adjacent sibling,
           since siblings (unlike parent/child) render correctly. */
        div:has(#delete-marker) + div button {
            background-color: #dc3545 !important;
            color: #ffffff !important;
            border: 1px solid #dc3545 !important;
        }

        div:has(#delete-marker) + div button:hover {
            background-color: #b02a37 !important;
            border-color: #b02a37 !important;
        }

        
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Resolve Selected Memory
# ============================================================

selected_memory_id = st.session_state.get("selected_memory_id")

if not selected_memory_id:
    selected_memory_id = st.query_params.get("memory_id")

if not selected_memory_id:
    st.warning("No memory was selected.")

    if st.button("← Back to Dashboard", type="primary"):
        st.switch_page("app.py")

    st.stop()


# ============================================================
# Load Memory
# ============================================================

memory = get_memory_by_id(
    memory_id=selected_memory_id,
    user_id=USER_ID,
)

if memory is None:
    st.error("This memory could not be found.")

    if st.button("← Back to Dashboard", type="primary"):
        st.switch_page("app.py")

    st.stop()


# ============================================================
# Back Button
# ============================================================

if st.button("← Back to Dashboard", key="back_to_dashboard"):
    st.session_state.pop("selected_memory_id", None)
    st.query_params.clear()
    st.switch_page("app.py")


# ============================================================
# Title (no date here — it already lives in Metadata)
# ============================================================

icon = {
    "image": "🖼️",
    "document": "📄",
    "audio": "🎙️",
}.get(memory["memory_type"], "🧠")

type_label = {
    "image": "Photo",
    "document": "Document",
    "audio": "Voice",
}.get(memory["memory_type"], memory["memory_type"].title())

st.markdown(f"## {icon} {memory['title']}")


# ============================================================
# Metadata
# ============================================================

with st.container(border=True):
    st.subheader("Metadata")

    meta_col1, meta_col2, meta_col3 = st.columns(3)

    with meta_col1:
        st.write(f"**Type:** {type_label}")

    with meta_col2:
        st.write(f"**Upload date:** {memory['created_at']}")

    with meta_col3:
        st.write(f"**File name:** {memory['file_name']}")


# ============================================================
# Original Content  +  Summary
# Both outer boxes use the SAME native `height=` value, so they
# are always identical regardless of content length. The Summary
# box additionally nests a smaller fixed-height container so only
# its value (not the "Summary" heading) scrolls.
# ============================================================

content_col, summary_col = st.columns([1, 1])

file_key = memory.get("original_file_key") or memory.get("file_path")
memory_type = memory["memory_type"]
file_url = generate_signed_url(file_key) if file_key else None
download_url = generate_signed_url(file_key, download_filename=memory["file_name"]) if file_key else None
safe_file_url = html.escape(file_url, quote=True) if file_url else None
safe_download_url = html.escape(download_url, quote=True) if download_url else None
safe_file_name = html.escape(memory["file_name"])

with content_col:
    with st.container(border=True, height=BOX_HEIGHT, key="memory_content_box"):
        if not file_url:
            st.error("The original file is unavailable.")
        elif memory_type == "image":
            st.image(file_url, use_container_width=True)
        elif memory_type == "audio":
            AUDIO_WIDGET_HEIGHT = 110
            spacer_height = max((BOX_HEIGHT - AUDIO_WIDGET_HEIGHT) // 2, 1)
            st.container(height=spacer_height, border=False, key="audio_spacer_top")
            st.audio(file_url)
            st.container(height=spacer_height, border=False, key="audio_spacer_bottom")
        elif memory_type == "document":
            mime_type, _ = mimetypes.guess_type(memory["file_name"])
            mime_type = mime_type or "application/octet-stream"
            if mime_type == "application/pdf":
                st.markdown(
                    f"""
                    <a href="{safe_download_url}" target="_blank" rel="noopener noreferrer" class="doc-preview-link">
                        <iframe src="{safe_file_url}"
                            style="width:100%; height:380px; border:none; border-radius:8px;">
                        </iframe>
                        <div style="text-align:center; margin-top:6px; font-size:0.85rem; opacity:0.75;">
                            📄 {safe_file_name} — click to download
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <a href="{safe_download_url}" target="_blank" rel="noopener noreferrer" class="doc-preview-link">
                        <div class="doc-preview-box">
                            <div style="font-size:2.5rem;">📄</div>
                            <div style="margin-top:8px;">{safe_file_name}</div>
                            <div style="font-size:0.8rem; opacity:0.65;">Click to download</div>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No original file is available for this memory type.")

with summary_col:
    with st.container(border=True, height=BOX_HEIGHT, key="memory_summary_box"):
        st.subheader("Summary")

        if memory["summary"]:
            with st.container(height=SUMMARY_INNER_HEIGHT, key="memory_summary_inner"):
                st.write(memory["summary"])
        else:
            # Keep the action outside the nested scrollable summary body.
            # This ensures the button remains visible and directly clickable.
            st.caption("No summary has been generated yet.")

            if st.button(
                "Generate Summary",
                type="primary",
                key=f"generate_summary_{memory['id']}",
            ):
                # Do not generate a summary until background ingestion and
                # indexing have completed successfully for this memory.
                connection = get_connection()
                try:
                    preparation_job = connection.execute(
                        """
                        SELECT status, current_stage
                        FROM preparation_jobs
                        WHERE memory_id = ? AND user_id = ?
                        ORDER BY created_at DESC, id DESC
                        LIMIT 1
                        """,
                        (memory["id"], USER_ID),
                    ).fetchone()
                finally:
                    connection.close()

                if preparation_job is not None and (
                    preparation_job["status"] != "succeeded"
                    or preparation_job["current_stage"] != "complete"
                ):
                    if preparation_job["status"] == "failed":
                        st.warning(
                            "Knowledge base preparation failed for this memory. "
                            "Please resolve the ingestion issue and try again "
                            "after preparation has completed."
                        )
                    else:
                        st.info(
                            "Memory ingestion and indexing are still in progress. "
                            "Please try again once knowledge base preparation is completed."
                        )
                else:
                    try:
                        with st.spinner("Generating title and summary..."):
                            generate_memory_summary(
                                memory_id=memory["id"],
                                user_id=USER_ID,
                            )

                        st.toast(
                            "AI title and summary generated successfully.",
                            icon="✅",
                        )
                        st.rerun()

                    except Exception as error:
                        st.toast(str(error), icon="❌")


# ============================================================
# Edit Title / Summary
# ============================================================

with st.expander("Edit Title / Summary"):
    with st.form(key=f"edit_memory_form_{memory['id']}"):

        edited_title = st.text_input(
            "Title",
            value=memory["title"],
        )

        edited_summary = st.text_area(
            "Summary",
            value=(memory["summary"] or ""),
            height=180,
        )

        save_changes = st.form_submit_button(
            "Save Changes",
            type="primary",
        )

        if save_changes:
            if not edited_title.strip():
                st.error("Title cannot be empty.")
            else:
                try:
                    update_memory(
                        memory_id=memory["id"],
                        user_id=USER_ID,
                        title=edited_title.strip(),
                        summary=edited_summary.strip(),
                    )

                    st.toast("Memory updated successfully.", icon="✅")
                    st.rerun()

                except Exception as error:
                    st.toast(str(error), icon="❌")


# ============================================================
# Delete Memory (just a red button, nothing else)
# ============================================================

st.markdown('<span id="delete-marker"></span>', unsafe_allow_html=True)

if st.button("🗑 Delete Memory", key=f"delete_memory_{memory['id']}"):
    try:
        delete_memory(
            memory_id=memory["id"],
            user_id=USER_ID,
        )

        st.session_state.pop("selected_memory_id", None)
        st.query_params.clear()

        st.toast("Memory deleted successfully.", icon="✅")
        st.switch_page("app.py")

    except Exception as error:
        st.toast(str(error), icon="❌")


# ============================================================
# Ask Question
# ============================================================

with st.container(border=True):
    st.subheader("Ask Question")

    st.write(
        "Ask questions about this memory using the "
        "Memory Vault conversational assistant."
    )

    if st.button(
        "Ask Question",
        type="primary",
        key=f"memory_detail_ask_question_{memory['id']}",
    ):
        from agent.graph import create_thread_id

        st.query_params["scope"] = "memory"
        st.query_params["memory_id"] = memory["id"]
        st.query_params["thread_id"] = create_thread_id(USER_ID)

        st.switch_page("pages/chat.py")