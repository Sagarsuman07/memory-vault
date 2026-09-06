import streamlit as st

from config.settings import settings

from database.database import initialize_database

from memory.memory_service import (
    create_new_memory,
    list_memories,
    update_memory,
    delete_memory,
    answer_question,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Memory Vault",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# Initialize Application
# ============================================================

initialize_database()

USER_ID = settings.DEMO_USER_ID


# ============================================================
# Header
# ============================================================

st.title(
    "🧠 Memory Vault"
)

st.write(
    "Your personal AI memory assistant."
)

st.divider()


# ============================================================
# Upload Memory
# ============================================================

st.header(
    "Upload a Memory"
)


uploaded_file = st.file_uploader(
    "Choose a memory file",
    type=[
        # Documents
        "pdf",
        "docx",
        "txt",

        # Images
        "jpg",
        "jpeg",
        "png",
        "webp",
        "gif",

        # Audio
        "mp3",
        "wav",
        "m4a",
        "mpeg",
        "mpga",
        "webm",
        "ogg",
        "flac",
    ],
)


title = st.text_input(
    "Memory Title",
    placeholder="e.g. Delhi Flight Ticket",
)


if st.button(
    "Save Memory",
    type="primary"
):

    if uploaded_file is None:

        st.warning(
            "Please select a memory file."
        )

    elif not title.strip():

        st.warning(
            "Please enter a memory title."
        )

    else:

        try:

            memory = create_new_memory(
                uploaded_file=uploaded_file,
                title=title,
                user_id=USER_ID,
            )


            st.success(
                f"Memory '{memory.title}' "
                "saved successfully."
            )


            st.rerun()


        except ValueError as error:

            st.error(
                str(error)
            )


        except Exception as error:

            st.error(
                f"Something went wrong: {error}"
            )


st.divider()


# ============================================================
# Memory Dashboard
# ============================================================

st.header(
    "My Memories"
)


memories = list_memories(
    USER_ID
)


if not memories:

    st.info(
        "You haven't uploaded any memories yet."
    )


else:

    for memory in memories:

        with st.container(
            border=True
        ):

            # -----------------------------------------
            # Icon
            # -----------------------------------------

            if memory["memory_type"] == "document":

                icon = "📄"

            elif memory["memory_type"] == "image":

                icon = "🖼️"

            elif memory["memory_type"] == "audio":

                icon = "🎵"

            else:

                icon = "🧠"


            # -----------------------------------------
            # Title
            # -----------------------------------------

            st.subheader(
                f"{icon} {memory['title']}"
            )


            # -----------------------------------------
            # Metadata
            # -----------------------------------------

            col1, col2, col3 = st.columns(3)


            with col1:

                st.write(
                    f"**Type:** "
                    f"{memory['memory_type'].upper()}"
                )


            with col2:

                st.write(
                    f"**File:** "
                    f"{memory['file_name']}"
                )


            with col3:

                st.write(
                    f"**Created:** "
                    f"{memory['created_at']}"
                )


            # -----------------------------------------
            # Summary
            # -----------------------------------------

            if memory["summary"]:

                st.write(
                    f"**Summary:** "
                    f"{memory['summary']}"
                )

            else:

                st.write(
                    "**Summary:** Not available"
                )


            # -----------------------------------------
            # Memory ID
            # -----------------------------------------

            st.write(
                f"**Memory ID:** "
                f"`{memory['id']}`"
            )


            # -----------------------------------------
            # User ID
            # -----------------------------------------

            st.write(
                f"**User ID:** "
                f"`{memory['user_id']}`"
            )


            # -----------------------------------------
            # File Path
            # -----------------------------------------

            st.write(
                f"**File Path:** "
                f"`{memory['file_path']}`"
            )


            # -----------------------------------------
            # Extracted Content
            # -----------------------------------------

            with st.expander(
                "View Extracted Content"
            ):

                st.text(
                    memory["extracted_text"]
                )


            # -----------------------------------------
            # Edit Memory
            # -----------------------------------------

            with st.expander(
                "Edit Memory"
            ):

                with st.form(
                    key=f"edit_form_{memory['id']}"
                ):

                    edited_title = st.text_input(
                        "Title",
                        value=memory["title"],
                    )


                    edited_summary = st.text_area(
                        "Summary",
                        value=memory["summary"] or "",
                    )


                    save_changes = st.form_submit_button(
                        "Save Changes"
                    )


                    if save_changes:

                        try:

                            update_memory(
                                memory_id=memory["id"],
                                user_id=USER_ID,
                                title=edited_title,
                                summary=edited_summary,
                            )


                            st.success(
                                "Memory updated successfully."
                            )


                            st.rerun()


                        except ValueError as error:

                            st.error(
                                str(error)
                            )


                        except Exception as error:

                            st.error(
                                f"Something went wrong: {error}"
                            )


            # -----------------------------------------
            # Delete Memory
            # -----------------------------------------

            if st.button(
                "Delete Memory",
                key=f"delete_{memory['id']}"
            ):

                try:

                    delete_memory(
                        memory_id=memory["id"],
                        user_id=USER_ID,
                    )


                    st.success(
                        "Memory deleted successfully."
                    )


                    st.rerun()


                except ValueError as error:

                    st.error(
                        str(error)
                    )


                except Exception as error:

                    st.error(
                        f"Something went wrong: {error}"
                    )


# ============================================================
# Ask Your Memories
# ============================================================

st.divider()

st.header(
    "Ask Your Memories"
)


# ============================================================
# Retrieval Scope
# ============================================================

search_scope = st.radio(
    "Search Scope",
    options=[
        "All Memories",
        "Specific Memory",
    ],
    horizontal=True,
)


selected_memory_id = None


# ============================================================
# Specific Memory Selection
# ============================================================

if search_scope == "Specific Memory":

    if not memories:

        st.info(
            "You don't have any memories to search."
        )

    else:

        memory_options = {
            memory["title"]: memory["id"]
            for memory in memories
        }


        selected_memory_title = st.selectbox(
            "Select a memory",
            options=list(
                memory_options.keys()
            ),
        )


        selected_memory_id = memory_options[
            selected_memory_title
        ]


# ============================================================
# Question
# ============================================================

question = st.text_input(
    "Ask a question about your memories",
    placeholder="e.g. What is the price of the iPhone?",
)


# ============================================================
# Ask Button
# ============================================================

if st.button(
    "Ask",
    type="primary",
    key="ask_memories",
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    elif (
        search_scope == "Specific Memory"
        and selected_memory_id is None
    ):

        st.warning(
            "Please select a memory."
        )

    else:

        try:

            result = answer_question(
                question=question,
                user_id=USER_ID,
                memory_id=selected_memory_id,
            )


            # -----------------------------------------
            # Answer
            # -----------------------------------------

            st.subheader(
                "Answer"
            )


            st.write(
                result["answer"]
            )


            # -----------------------------------------
            # Sources
            # -----------------------------------------

            if result["grounded"]:

                st.subheader(
                    "Sources"
                )


                for source in result["sources"]:

                    st.write(
                        f"- Memory ID: "
                        f"`{source['memory_id']}`"
                    )

                    st.write(
                        f"  Chunk: "
                        f"`{source['chunk_index']}`"
                    )

                    st.write(
                        f"  Distance: "
                        f"`{source['distance']:.4f}`"
                    )


        except Exception as error:

            st.error(
                f"Something went wrong while answering your question: {error}"
            )