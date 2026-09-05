import streamlit as st

from config.settings import settings

from database.database import (
    initialize_database,
)

from database.repositories import (
    get_all_memories,
)

from memory.memory_service import (
    create_new_memory,
    delete_memory,
)


st.set_page_config(
    page_title="Memory Vault",
    page_icon="🧠",
    layout="wide",
)


# Initialize database.
initialize_database()


USER_ID = settings.DEMO_USER_ID


st.title("🧠 Memory Vault")

st.write(
    "Your personal AI memory assistant."
)

st.divider()


# ============================================================
# Upload Memory
# ============================================================

st.header("Upload a Memory")


uploaded_file = st.file_uploader(
    "Choose a memory file",
    type=[
        "pdf",
        "docx",
        "txt",
        "jpg",
        "jpeg",
        "png",
        "webp",
        "gif",
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
    type="primary",
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
                title=title.strip(),
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
# Memories
# ============================================================

st.header("My Memories")


memories = get_all_memories(
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

            # Choose an icon based on memory type.
            if memory["memory_type"] == "document":
                icon = "📄"

            elif memory["memory_type"] == "image":
                icon = "🖼️"

            elif memory["memory_type"] == "audio":
                icon = "🎵"

            else:
                icon = "🧠"

            st.subheader(
                f"{icon} {memory['title']}"
            )

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

            with st.expander(
                "Memory Details"
            ):

                st.write(
                    f"**Memory ID:** "
                    f"`{memory['id']}`"
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

            if st.button(
                "Delete Memory",
                key=f"delete_{memory['id']}",
            ):

                deleted = delete_memory(
                    memory_id=memory["id"],
                    user_id=USER_ID,
                )

                if deleted:

                    st.success(
                        "Memory deleted."
                    )

                    st.rerun()

                else:

                    st.error(
                        "Memory could not be found."
                    )