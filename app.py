import streamlit as st

from agent.graph import (
    run_langgraph_agent,
    create_thread_id,
    get_default_thread_id,
    get_thread_state,
)

from config.settings import settings

from database.database import initialize_database

from memory.memory_service import (
    list_memories,
    update_memory,
    delete_memory,
    answer_question,
    generate_memory_summary,
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


# ============================================================
# Add Memory Entry Point
# ============================================================

st.subheader(
    "Add a new memory"
)

st.write(
    "Save documents, photos, or voice recordings "
    "to your personal memory vault."
)

if st.button(
    "+ Add Memory",
    type="primary",
    key="open_add_memory",
):

    st.switch_page(
        "pages/add_memory.py"
    )


# ============================================================
# Dashboard
# ============================================================

st.divider()

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

            # ------------------------------------------------
            # Memory Icon
            # ------------------------------------------------

            if (
                memory["memory_type"]
                == "document"
            ):

                icon = "📄"

            elif (
                memory["memory_type"]
                == "image"
            ):

                icon = "🖼️"

            elif (
                memory["memory_type"]
                == "audio"
            ):

                icon = "🎵"

            else:

                icon = "🧠"

            # ------------------------------------------------
            # Title
            # ------------------------------------------------

            st.subheader(
                f"{icon} {memory['title']}"
            )

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            col1, col2, col3 = st.columns(
                3
            )

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

            # ------------------------------------------------
            # Summary
            # ------------------------------------------------

            if memory["summary"]:

                st.write(
                    f"**Summary:** "
                    f"{memory['summary']}"
                )

            else:

                st.write(
                    "**Summary:** Not available"
                )

            # ------------------------------------------------
            # Generate Summary
            # ------------------------------------------------

            if st.button(
                "Generate Summary",
                key=(
                    f"generate_summary_"
                    f"{memory['id']}"
                ),
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
                        (
                            "AI title and summary "
                            "generated successfully."
                        ),
                        icon="✅",
                    )

                    st.rerun()

                except ValueError as error:

                    st.toast(
                        str(error),
                        icon="❌",
                    )

                except Exception as error:

                    st.toast(
                        (
                            "Something went wrong "
                            "while generating the "
                            f"summary: {error}"
                        ),
                        icon="❌",
                    )

            # ------------------------------------------------
            # Extracted Content
            # ------------------------------------------------

            with st.expander(
                "View Extracted Content"
            ):

                st.text(
                    memory["extracted_text"]
                )

            # ------------------------------------------------
            # Edit Memory
            # ------------------------------------------------

            with st.expander(
                "Edit Memory"
            ):

                with st.form(
                    key=(
                        f"edit_form_"
                        f"{memory['id']}"
                    )
                ):

                    edited_title = st.text_input(
                        "Title",
                        value=memory["title"],
                    )

                    edited_summary = st.text_area(
                        "Summary",
                        value=(
                            memory["summary"]
                            or ""
                        ),
                    )

                    save_changes = (
                        st.form_submit_button(
                            "Save Changes"
                        )
                    )

                    if save_changes:

                        try:

                            update_memory(
                                memory_id=memory["id"],
                                user_id=USER_ID,
                                title=edited_title,
                                summary=edited_summary,
                            )

                            st.toast(
                                (
                                    "Memory updated "
                                    "successfully."
                                ),
                                icon="✅",
                            )

                            st.rerun()

                        except ValueError as error:

                            st.toast(
                                str(error),
                                icon="❌",
                            )

                        except Exception as error:

                            st.toast(
                                (
                                    "Something went "
                                    "wrong: "
                                    f"{error}"
                                ),
                                icon="❌",
                            )

            # ------------------------------------------------
            # Delete Memory
            # ------------------------------------------------

            if st.button(
                "Delete Memory",
                key=(
                    f"delete_"
                    f"{memory['id']}"
                ),
            ):

                try:

                    delete_memory(
                        memory_id=memory["id"],
                        user_id=USER_ID,
                    )

                    st.toast(
                        (
                            "Memory deleted "
                            "successfully."
                        ),
                        icon="✅",
                    )

                    st.rerun()

                except ValueError as error:

                    st.toast(
                        str(error),
                        icon="❌",
                    )

                except Exception as error:

                    st.toast(
                        (
                            "Something went wrong: "
                            f"{error}"
                        ),
                        icon="❌",
                    )


# ============================================================
# Grounded RAG
# ============================================================

st.divider()

st.header(
    "Ask Your Memories"
)

search_scope = st.radio(
    "Search Scope",
    options=[
        "All Memories",
        "Specific Memory",
    ],
    horizontal=True,
)

selected_memory_id = None


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

        selected_memory_title = (
            st.selectbox(
                "Select a memory",
                options=list(
                    memory_options.keys()
                ),
            )
        )

        selected_memory_id = (
            memory_options[
                selected_memory_title
            ]
        )


question = st.text_input(
    "Ask a question about your memories",
    placeholder=(
        "e.g. What is the price of the iPhone?"
    ),
)


if st.button(
    "Ask",
    type="primary",
    key="ask_memories",
):

    if not question.strip():

        st.toast(
            "Please enter a question.",
            icon="⚠️",
        )

    elif (
        search_scope == "Specific Memory"
        and selected_memory_id is None
    ):

        st.toast(
            "Please select a memory.",
            icon="⚠️",
        )

    else:

        try:

            result = answer_question(
                question=question,
                user_id=USER_ID,
                memory_id=selected_memory_id,
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

                for source in result[
                    "sources"
                ]:

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

            st.toast(
                (
                    "Something went wrong "
                    "while answering your "
                    f"question: {error}"
                ),
                icon="❌",
            )


# ============================================================
# LangGraph Agent
# ============================================================

st.divider()

st.header(
    "Ask Memory Vault LangGraph Agent"
)

st.write(
    "This agent maintains short-term conversational "
    "memory using LangGraph threads and a persistent "
    "SQLite checkpointer."
)


# ============================================================
# Thread Initialization
# ============================================================

if (
    "langgraph_thread_id"
    not in st.session_state
):

    st.session_state.langgraph_thread_id = (
        get_default_thread_id(
            USER_ID
        )
    )


# ============================================================
# New Conversation
# ============================================================

if st.button(
    "New Conversation",
    key="new_langgraph_conversation",
):

    st.session_state.langgraph_thread_id = (
        create_thread_id(
            USER_ID
        )
    )

    st.rerun()


# ============================================================
# Current Thread
# ============================================================

st.caption(
    (
        "Current conversation: "
        f"`{st.session_state.langgraph_thread_id}`"
    )
)


# ============================================================
# Load Thread State
# ============================================================

try:

    thread_state = get_thread_state(
        user_id=USER_ID,
        thread_id=(
            st.session_state
            .langgraph_thread_id
        ),
    )

    conversation_messages = (
        thread_state.values.get(
            "messages",
            [],
        )
    )

except Exception:

    conversation_messages = []


# ============================================================
# Display Conversation
# ============================================================

for message in conversation_messages:

    message_type = getattr(
        message,
        "type",
        "",
    )

    if message_type == "human":

        with st.chat_message(
            "user"
        ):

            st.write(
                message.content
            )

    elif message_type == "ai":

        content = message.content

        if content:

            with st.chat_message(
                "assistant"
            ):

                st.write(
                    content
                )


# ============================================================
# LangGraph Chat Input
# ============================================================

langgraph_question = st.chat_input(
    "Ask Memory Vault anything..."
)


if langgraph_question:

    try:

        with st.spinner(
            "Memory Vault is thinking..."
        ):

            run_langgraph_agent(
                question=langgraph_question,
                user_id=USER_ID,
                thread_id=(
                    st.session_state
                    .langgraph_thread_id
                ),
            )

        st.rerun()

    except Exception as error:

        st.toast(
            (
                "Something went wrong while "
                "running the LangGraph agent: "
                f"{error}"
            ),
            icon="❌",
        )