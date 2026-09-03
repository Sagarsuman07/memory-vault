import streamlit as st

from config.settings import settings

from database.database import (
    initialize_database
)

from database.repositories import (
    get_all_memories
)

from memory.memory_service import (
    create_new_memory,
    delete_memory
)


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Memory Vault",
    page_icon="🧠",
    layout="wide",
)


# ==================================================
# Application Initialization
# ==================================================

initialize_database()

USER_ID = settings.DEMO_USER_ID


# ==================================================
# Header
# ==================================================

st.title(
    "🧠 Memory Vault"
)

st.write(
    "Your personal AI memory assistant."
)

st.divider()


# ==================================================
# Upload Section
# ==================================================

st.header(
    "Upload a Memory"
)


uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=[
        "pdf",
        "txt"
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

    # ----------------------------------------------
    # Validate title/file from UI
    # ----------------------------------------------

    if uploaded_file is None:

        st.warning(
            "Please select a PDF or TXT file."
        )

    elif not title.strip():

        st.warning(
            "Please enter a memory title."
        )

    else:

        try:

            # --------------------------------------
            # Create memory
            # --------------------------------------

            memory = create_new_memory(
                uploaded_file=uploaded_file,
                title=title.strip(),
                user_id=USER_ID,
            )


            st.success(
                f"Memory '{memory.title}' "
                "saved successfully."
            )


            # Refresh dashboard

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


# ==================================================
# Memory Dashboard
# ==================================================

st.header(
    "My Memories"
)


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

            # --------------------------------------
            # Memory Title
            # --------------------------------------

            st.subheader(
                f"📄 {memory['title']}"
            )


            # --------------------------------------
            # Metadata columns
            # --------------------------------------

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


            # --------------------------------------
            # Memory Details
            # --------------------------------------

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


            # --------------------------------------
            # Delete Memory
            # --------------------------------------

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