import os

import streamlit as st

from config.settings import settings
from database.database import initialize_database

from memory.memory_service import (
    get_memory_by_id,
    generate_memory_summary,
    update_memory,
    delete_memory,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Memory Detail",
    page_icon="🧠",
    layout="wide",
)

initialize_database()

USER_ID = settings.DEMO_USER_ID


# ============================================================
# Resolve Selected Memory
# ============================================================

selected_memory_id = st.session_state.get(
    "selected_memory_id"
)

if not selected_memory_id:
    selected_memory_id = st.query_params.get(
        "memory_id"
    )

if not selected_memory_id:
    st.warning("No memory was selected.")

    if st.button(
        "← Back to Dashboard",
        type="primary",
    ):
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

    if st.button(
        "← Back to Dashboard",
        type="primary",
    ):
        st.switch_page("app.py")

    st.stop()


# ============================================================
# Header
# ============================================================

if st.button(
    "← Back to Dashboard",
    key="back_to_dashboard",
):
    st.session_state.pop(
        "selected_memory_id",
        None,
    )
    st.query_params.clear()
    st.switch_page("app.py")


icon = {
    "image": "🖼️",
    "document": "📄",
    "audio": "🎙️",
}.get(
    memory["memory_type"],
    "🧠",
)

type_label = {
    "image": "Photo",
    "document": "Document",
    "audio": "Voice",
}.get(
    memory["memory_type"],
    memory["memory_type"].title(),
)


st.title(
    f"{icon} {memory['title']}"
)

st.caption(
    f"{type_label} • {memory['created_at']}"
)


# ============================================================
# Metadata
# ============================================================

st.subheader("Metadata")

meta_col1, meta_col2, meta_col3 = st.columns(3)

with meta_col1:
    st.write(f"**Type:** {type_label}")

with meta_col2:
    st.write(f"**Upload date:** {memory['created_at']}")

with meta_col3:
    st.write(f"**File name:** {memory['file_name']}")


st.divider()


# ============================================================
# Full Content
# ============================================================

st.subheader("Full Content")

file_path = memory["file_path"]
memory_type = memory["memory_type"]


if memory_type == "image":
    if os.path.exists(file_path):
        st.image(
            file_path,
            use_container_width=True,
        )
    else:
        st.error("The original image file is missing.")

elif memory_type == "audio":
    if os.path.exists(file_path):
        st.audio(file_path)
    else:
        st.error("The original audio file is missing.")

    if memory["extracted_text"]:
        st.markdown("**Transcript**")
        st.text_area(
            "Transcript",
            value=memory["extracted_text"],
            height=250,
            disabled=True,
            label_visibility="collapsed",
        )

elif memory_type == "document":
    # Streamlit does not provide a universal PDF/DOCX viewer.
    # Present the extracted document content in a readable native component.
    if memory["extracted_text"]:
        st.text_area(
            "Document content",
            value=memory["extracted_text"],
            height=500,
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        st.info("No extracted document content is available.")

else:
    st.text_area(
        "Memory content",
        value=memory["extracted_text"] or "",
        height=400,
        disabled=True,
        label_visibility="collapsed",
    )


st.divider()


# ============================================================
# Summary
# ============================================================

st.subheader("Summary")

summary_col1, summary_col2 = st.columns(
    [1, 1]
)

with summary_col1:
    st.markdown("**Title**")
    st.write(memory["title"])

with summary_col2:
    st.markdown("**Summary**")

    if memory["summary"]:
        st.write(memory["summary"])
    else:
        st.caption(
            "No summary has been generated yet."
        )


if st.button(
    "Generate Summary",
    type="primary",
    key=f"generate_summary_{memory['id']}",
):
    try:
        with st.spinner(
            "Generating title and summary..."
        ):
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
        st.toast(
            str(error),
            icon="❌",
        )


# ============================================================
# Edit Title / Summary
# ============================================================

with st.expander(
    "Edit Title / Summary"
):
    with st.form(
        key=f"edit_memory_form_{memory['id']}"
    ):
        edited_title = st.text_input(
            "Title",
            value=memory["title"],
        )

        edited_summary = st.text_area(
            "Summary",
            value=memory["summary"] or "",
            height=180,
        )

        save_changes = st.form_submit_button(
            "Save Changes",
            type="primary",
        )

        if save_changes:
            if not edited_title.strip():
                st.error(
                    "Title cannot be empty."
                )
            else:
                try:
                    update_memory(
                        memory_id=memory["id"],
                        user_id=USER_ID,
                        title=edited_title.strip(),
                        summary=edited_summary.strip(),
                    )

                    st.toast(
                        "Memory updated successfully.",
                        icon="✅",
                    )

                    st.rerun()

                except Exception as error:
                    st.toast(
                        str(error),
                        icon="❌",
                    )


st.caption(
    "Editing the title or summary updates SQLite only. "
    "Original extracted content and RAG vectors are unchanged."
)


# ============================================================
# Delete Memory
# ============================================================

st.divider()

with st.expander("Danger Zone"):
    st.warning(
        "Deleting this memory removes its database record, "
        "original file, and associated vector data."
    )

    if st.button(
        "Delete Memory",
        key=f"delete_memory_{memory['id']}",
    ):
        try:
            delete_memory(
                memory_id=memory["id"],
                user_id=USER_ID,
            )

            st.session_state.pop(
                "selected_memory_id",
                None,
            )
            st.query_params.clear()

            st.toast(
                "Memory deleted successfully.",
                icon="✅",
            )

            st.switch_page("app.py")

        except Exception as error:
            st.toast(
                str(error),
                icon="❌",
            )


# ============================================================
# Phase 15 Placeholder
# ============================================================

st.divider()

st.subheader("Ask Question")

st.info(
    "Memory-specific chat will be implemented in Phase 15."
)
