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
    delete_memories,
    load_demo_memories,
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

initialize_database()

USER_ID = settings.DEMO_USER_ID


# ============================================================
# Session State
# ============================================================

if "edit_memories_mode" not in st.session_state:
    st.session_state.edit_memories_mode = False

if "selected_memory_ids" not in st.session_state:
    st.session_state.selected_memory_ids = set()

if "langgraph_thread_id" not in st.session_state:
    st.session_state.langgraph_thread_id = (
        get_default_thread_id(USER_ID)
    )


# ============================================================
# Helpers
# ============================================================

def get_type_icon(memory_type: str) -> str:
    icons = {
        "image": "🖼️",
        "document": "📄",
        "audio": "🎙️",
    }

    return icons.get(memory_type, "🧠")


def get_type_label(memory_type: str) -> str:
    labels = {
        "image": "Photo",
        "document": "Document",
        "audio": "Voice",
    }

    return labels.get(
        memory_type,
        memory_type.title(),
    )


def format_date(created_at: str) -> str:
    try:
        return created_at[:10]
    except Exception:
        return created_at


def get_summary_preview(
    memory,
    max_length: int = 120,
):
    # sqlite3.Row supports [] access, not .get()
    summary = memory["summary"]

    if not summary:
        return None

    summary = summary.strip()

    if len(summary) <= max_length:
        return summary

    return summary[: max_length - 3] + "..."


def filter_by_type(
    memories,
    memory_type=None,
):
    if memory_type is None:
        return list(memories)

    return [
        memory
        for memory in memories
        if memory["memory_type"] == memory_type
    ]


def filter_by_title(
    memories,
    search_text: str,
):
    if not search_text:
        return list(memories)

    search_text = search_text.strip().lower()

    if not search_text:
        return list(memories)

    return [
        memory
        for memory in memories
        if search_text in memory["title"].lower()
    ]


def open_memory(memory_id: str):
    st.session_state.selected_memory_id = memory_id

    st.query_params["memory_id"] = memory_id

    st.switch_page(
        "pages/memory_detail.py"
    )


def clear_memory_selection():
    st.session_state.selected_memory_ids = set()


def sync_memory_selection(
    memory_id: str,
    checkbox_key: str,
):
    selected_ids = set(
        st.session_state.get(
            "selected_memory_ids",
            set(),
        )
    )

    if st.session_state.get(
        checkbox_key,
        False,
    ):
        selected_ids.add(memory_id)
    else:
        selected_ids.discard(memory_id)

    st.session_state.selected_memory_ids = (
        selected_ids
    )


# ============================================================
# Compact Dashboard UI (sticky behavior removed)
# ============================================================

st.markdown(
    """
    <style>

    /* --------------------------------------------------------
       Reduce unused space at the top
       Keeps content clear of Streamlit's fixed toolbar
       -------------------------------------------------------- */

    [data-testid="stMainBlockContainer"] {
        padding-top: 3.75rem;
    }


    /* --------------------------------------------------------
       Dashboard header container
       (sticky positioning removed — now scrolls normally)
       -------------------------------------------------------- */

    .st-key-dashboard-sticky-header {
        padding-top: 0.20rem;
        padding-bottom: 0.45rem;

        margin-bottom: 0.35rem;

        border-bottom: 1px solid
            rgba(128, 128, 128, 0.18);
    }


    /* --------------------------------------------------------
       Dashboard title
       -------------------------------------------------------- */

    .st-key-dashboard-sticky-header h1 {
        margin-top: 0;
        margin-bottom: 0.10rem;

        padding-top: 0;
        padding-bottom: 0;

        line-height: 1.20;
    }


    /* --------------------------------------------------------
       Subtitle
       -------------------------------------------------------- */

    .st-key-dashboard-sticky-header
    [data-testid="stMarkdownContainer"] p {
        margin-top: 0.05rem;
        margin-bottom: 0.30rem;
    }


    /* --------------------------------------------------------
       Dashboard controls
       -------------------------------------------------------- */

    .st-key-dashboard-sticky-header
    [data-testid="stHorizontalBlock"] {
        gap: 0.60rem;
    }


    /* --------------------------------------------------------
       Delete button
       -------------------------------------------------------- */

    .st-key-dashboard-sticky-header
    button[kind="secondary"] {
        background-color: #dc2626;
        color: white;
        border-color: #dc2626;
    }


    .st-key-dashboard-sticky-header
    button[kind="secondary"]:hover:not(:disabled) {
        background-color: #b91c1c;
        border-color: #b91c1c;
        color: white;
    }


    /* --------------------------------------------------------
       Disabled delete button
       -------------------------------------------------------- */

    .st-key-dashboard-sticky-header
    button:disabled {
        opacity: 0.45;
        cursor: not-allowed;
    }


    /* --------------------------------------------------------
       Dashboard tabs
       -------------------------------------------------------- */

    .st-key-dashboard-sticky-header
    .stTabs [data-baseweb="tab-list"] {
        padding-top: 0.05rem;
        padding-bottom: 0.05rem;

        margin-top: 0.10rem;
    }


    /* --------------------------------------------------------
       Compact vertical spacing
       -------------------------------------------------------- */

    [data-testid="stMainBlockContainer"]
    [data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Load Memories
# IMPORTANT:
# This must happen before anything references "memories".
# ============================================================

memories = list_memories(USER_ID)


# ============================================================
# Remove Invalid Selections
# ============================================================

valid_memory_ids = {
    memory["id"]
    for memory in memories
}

st.session_state.selected_memory_ids = {
    memory_id
    for memory_id
    in st.session_state.selected_memory_ids
    if memory_id in valid_memory_ids
}


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("Memory Vault")


    # --------------------------------------------------------
    # Sidebar Add Memory
    # --------------------------------------------------------

    if st.button(
        "+ Add Memory",
        type="primary",
        use_container_width=True,
        key="sidebar_add_memory",
    ):
        st.switch_page(
            "pages/add_memory.py"
        )


    st.divider()


    # --------------------------------------------------------
    # Quick Search
    # --------------------------------------------------------

    st.subheader("Quick Search")

    search_text = st.text_input(
        "Search memories...",
        placeholder="Search by title",
        key="memory_title_search",
    )

    st.caption(
        "Title search only — no RAG or AI."
    )


    st.divider()


    # --------------------------------------------------------
    # Demo Memories
    # --------------------------------------------------------

    st.subheader("Demo")

    if st.button(
        "Load Demo Memories",
        use_container_width=True,
        key="load_demo_memories",
    ):
        try:

            with st.spinner(
                "Preparing demo memories..."
            ):
                created = load_demo_memories(
                    USER_ID
                )


            if created:

                st.toast(
                    (
                        f"{len(created)} "
                        "demo memory/memories loaded."
                    ),
                    icon="✅",
                )

            else:

                st.toast(
                    "Demo memories are already loaded.",
                    icon="ℹ️",
                )


            st.rerun()


        except Exception as error:

            st.error(
                (
                    "Demo memories could not "
                    f"be loaded: {error}"
                )
            )


# ============================================================
# Dashboard Header (no longer sticky)
# ============================================================

with st.container(
    key="dashboard_sticky_header"
):


    # ========================================================
    # Header
    # ========================================================

    st.title(
        "🧠 Memory Vault"
    )

    st.write(
        "Your personal AI memory assistant."
    )


    # ========================================================
    # Dashboard Controls
    # ========================================================

    control_col1, control_col2, control_col3 = (
        st.columns(
            [1.4, 1.4, 4.2]
        )
    )


    # --------------------------------------------------------
    # Add / Delete
    # --------------------------------------------------------

    with control_col1:

        if st.session_state.get(
            "edit_memories_mode",
            False,
        ):

            selected_count = len(
                st.session_state.get(
                    "selected_memory_ids",
                    set(),
                )
            )


            delete_clicked = st.button(
                f"🗑️ Delete ({selected_count})",
                type="secondary",
                use_container_width=True,
                disabled=(
                    selected_count == 0
                ),
                key="homepage_delete_memories",
            )


            if delete_clicked:

                try:

                    deleted_count = (
                        delete_memories(
                            memory_ids=list(
                                st.session_state.get(
                                    "selected_memory_ids",
                                    set(),
                                )
                            ),
                            user_id=USER_ID,
                        )
                    )


                    clear_memory_selection()

                    st.session_state.edit_memories_mode = (
                        False
                    )


                    st.toast(
                        (
                            f"{deleted_count} "
                            "memory/memories deleted."
                        ),
                        icon="✅",
                    )


                    st.rerun()


                except Exception as error:

                    st.toast(
                        (
                            "Selected memories "
                            "could not be deleted: "
                            f"{error}"
                        ),
                        icon="❌",
                    )


        else:

            if st.button(
                "+ Add Memory",
                type="primary",
                use_container_width=True,
                key="homepage_add_memory",
            ):

                st.switch_page(
                    "pages/add_memory.py"
                )


    # --------------------------------------------------------
    # Edit / Cancel
    # --------------------------------------------------------

    with control_col2:

        if st.session_state.get(
            "edit_memories_mode",
            False,
        ):

            if st.button(
                "Cancel",
                use_container_width=True,
                key="cancel_edit_memories",
            ):

                st.session_state.edit_memories_mode = (
                    False
                )

                clear_memory_selection()

                st.rerun()


        else:

            if st.button(
                "Edit Memories",
                use_container_width=True,
                key="edit_memories",
            ):

                st.session_state.edit_memories_mode = (
                    True
                )

                clear_memory_selection()

                st.rerun()


    # --------------------------------------------------------
    # Selection Status
    # --------------------------------------------------------

    with control_col3:

        if st.session_state.get(
            "edit_memories_mode",
            False,
        ):

            selected_count = len(
                st.session_state.get(
                    "selected_memory_ids",
                    set(),
                )
            )


            if selected_count:

                st.caption(
                    f"{selected_count} selected"
                )

            else:

                st.caption(
                    "Select memories to enable delete."
                )


    # ========================================================
    # Counts
    # ========================================================

    all_count = len(
        memories
    )


    photo_count = len(
        filter_by_type(
            memories,
            "image",
        )
    )


    document_count = len(
        filter_by_type(
            memories,
            "document",
        )
    )


    voice_count = len(
        filter_by_type(
            memories,
            "audio",
        )
    )


    # ========================================================
    # Dashboard Tabs
    # ========================================================

    all_tab, photo_tab, document_tab, voice_tab = (
        st.tabs(
            [
                f"All ({all_count})",
                f"Photos ({photo_count})",
                f"Documents ({document_count})",
                f"Voice ({voice_count})",
            ]
        )
    )


# ============================================================
# Memory Card Grid
# ============================================================

def render_memory_grid(
    tab_memories,
    tab_name: str,
):

    filtered_memories = filter_by_title(
        tab_memories,
        search_text,
    )


    if not filtered_memories:

        if search_text:

            st.info(
                "No memories match your title search."
            )

        else:

            st.info(
                f"No {tab_name.lower()} memories found."
            )

        return


    st.caption(
        (
            f"Showing {len(filtered_memories)} "
            f"{tab_name.lower()}."
        )
    )


    columns = st.columns(3)


    for index, memory in enumerate(
        filtered_memories
    ):

        column = columns[
            index % 3
        ]


        with column:

            # ------------------------------------------------
            # Native memory card
            # ------------------------------------------------

            with st.container(
                border=True
            ):

                icon = get_type_icon(
                    memory["memory_type"]
                )

                type_label = get_type_label(
                    memory["memory_type"]
                )


                st.markdown(
                    f"### {icon} {type_label}"
                )


                st.write(
                    f"**{memory['title']}**"
                )


                st.caption(
                    format_date(
                        memory["created_at"]
                    )
                )


                preview = (
                    get_summary_preview(
                        memory
                    )
                )


                if preview:

                    st.write(
                        preview
                    )

                else:

                    st.caption(
                        "Open memory to generate summary"
                    )


                # ------------------------------------------------
                # Selection checkbox
                # ------------------------------------------------

                if st.session_state.get(
                    "edit_memories_mode",
                    False,
                ):

                    checkbox_key = (
                        f"select_"
                        f"{tab_name}_"
                        f"{memory['id']}"
                    )


                    if checkbox_key not in (
                        st.session_state
                    ):

                        st.session_state[
                            checkbox_key
                        ] = (
                            memory["id"]
                            in st.session_state.get(
                                "selected_memory_ids",
                                set(),
                            )
                        )


                    st.checkbox(
                        "Select",
                        key=checkbox_key,
                        on_change=(
                            sync_memory_selection
                        ),
                        args=(
                            memory["id"],
                            checkbox_key,
                        ),
                    )


                # ------------------------------------------------
                # Open Memory
                # ------------------------------------------------

                if st.button(
                    "Open Memory →",
                    key=(
                        f"open_"
                        f"{tab_name}_"
                        f"{memory['id']}"
                    ),
                    use_container_width=True,
                ):

                    open_memory(
                        memory["id"]
                    )


# ============================================================
# Dashboard Tabs
# ============================================================

with all_tab:

    render_memory_grid(
        memories,
        "All",
    )


with photo_tab:

    render_memory_grid(
        filter_by_type(
            memories,
            "image",
        ),
        "Photos",
    )


with document_tab:

    render_memory_grid(
        filter_by_type(
            memories,
            "document",
        ),
        "Documents",
    )


with voice_tab:

    render_memory_grid(
        filter_by_type(
            memories,
            "audio",
        ),
        "Voice",
    )


# ============================================================
# Existing Grounded RAG
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
                        (
                            "- Memory ID: "
                            f"`{source['memory_id']}`"
                        )
                    )


                    st.write(
                        (
                            "  Chunk: "
                            f"`{source['chunk_index']}`"
                        )
                    )


                    st.write(
                        (
                            "  Distance: "
                            f"`{source['distance']:.4f}`"
                        )
                    )


        except Exception as error:

            st.toast(
                str(error),
                icon="❌",
            )


# ============================================================
# Existing LangGraph Agent
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
# New Conversation
# ============================================================

if st.button(
    "New Conversation",
    key="new_langgraph_conversation",
):

    st.session_state.langgraph_thread_id = (
        create_thread_id(USER_ID)
    )

    st.rerun()


st.caption(
    "Current conversation: "
    f"`{st.session_state.langgraph_thread_id}`"
)


# ============================================================
# Load Conversation
# ============================================================

try:

    thread_state = get_thread_state(
        user_id=USER_ID,
        thread_id=(
            st.session_state.langgraph_thread_id
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

        if message.content:

            with st.chat_message(
                "assistant"
            ):

                st.write(
                    message.content
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