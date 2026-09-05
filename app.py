import streamlit as st

from config.settings import settings

from database.database import (
    initialize_database
)

from memory.memory_service import (
    create_new_memory,
    list_memories,
    update_memory,
    delete_memory,
)


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
            # Memory Details
            # -----------------------------------------

            with st.expander(
                "Memory Details"
            ):

                st.write(
                    f"**Memory ID:** "
                    f"`{memory['id']}`"
                )


                st.write(
                    f"**User ID:** "
                    f"`{memory['user_id']}`"
                )


                st.write(
                    f"**File Path:** "
                    f"`{memory['file_path']}`"
                )


                st.write(
                    "**Extracted Content:**"
                )


                st.text(
                    memory["extracted_text"]
                )


            st.divider()


            # -----------------------------------------
            # Update Memory
            # -----------------------------------------

            with st.expander(
                "Edit Memory"
            ):

                with st.form(
                    key=f"edit_form_{memory['id']}"
                ):

                    updated_title = st.text_input(
                        "Title",
                        value=memory["title"],
                    )


                    updated_summary = st.text_area(
                        "Summary",
                        value=memory["summary"] or "",
                        height=120,
                    )


                    save_changes = st.form_submit_button(
                        "Save Changes"
                    )


                    if save_changes:

                        try:

                            update_memory(
                                memory_id=memory["id"],
                                user_id=USER_ID,
                                title=updated_title,
                                summary=updated_summary,
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
                key=f"delete_{memory['id']}",
            ):

                try:

                    deleted = delete_memory(
                        memory_id=memory["id"],
                        user_id=USER_ID,
                    )


                    if deleted:

                        st.success(
                            "Memory deleted successfully."
                        )

                        st.rerun()


                    else:

                        st.error(
                            "Memory could not be found."
                        )


                except Exception as error:

                    st.error(
                        f"Something went wrong: {error}"
                    )



# ============================================================
# Question and Answer
# ============================================================

st.divider()

st.header(
    "Ask Your Memories"
)


question = st.text_input(
    "Ask a question about your memories",
    placeholder="e.g. What is the price of the iPhone?"
)


if st.button(
    "Ask",
    type="primary"
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            result = answer_question(
                question=question,
                user_id=USER_ID,
            )


            st.subheader(
                "Answer"
            )

            st.write(
                result["answer"]
            )


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