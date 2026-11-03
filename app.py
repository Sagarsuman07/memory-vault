import streamlit as st

from agent.graph import (
    create_thread_id as create_chat_thread_id,
    initialize_checkpointer,
)

from config.settings import settings
from database.database import initialize_database
from memory.background_worker import start_background_worker
from rag.vector_store import get_client as initialize_qdrant

from memory.memory_service import (
    list_memories_page,
    count_memories_by_type,
    get_existing_memory_ids,
    delete_memories,
    load_demo_memories,
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
initialize_checkpointer()
initialize_qdrant()
start_background_worker()

USER_ID = settings.DEMO_USER_ID


# ============================================================
# Session State
# ============================================================

if "edit_memories_mode" not in st.session_state:
    st.session_state.edit_memories_mode = False

if "selected_memory_ids" not in st.session_state:
    st.session_state.selected_memory_ids = set()


# ============================================================
# Helpers
# ============================================================

def get_type_icon(
    memory_type: str,
) -> str:

    icons = {
        "image": "🖼️",
        "document": "📄",
        "audio": "🎙️",
    }

    return icons.get(
        memory_type,
        "🧠",
    )


def get_type_label(
    memory_type: str,
) -> str:

    labels = {
        "image": "Photo",
        "document": "Document",
        "audio": "Voice",
    }

    return labels.get(
        memory_type,
        memory_type.title(),
    )


def format_date(
    created_at: str,
) -> str:

    try:
        return created_at[:10]

    except Exception:
        return created_at


def get_summary_preview(
    memory,
    max_length: int = 120,
):

    summary = memory["summary"]

    if not summary:
        return None

    summary = summary.strip()

    if len(summary) <= max_length:
        return summary

    return (
        summary[: max_length - 3]
        + "..."
    )


def filter_by_type(
    memories,
    memory_type=None,
):

    if memory_type is None:
        return list(memories)

    return [
        memory
        for memory in memories
        if memory["memory_type"]
        == memory_type
    ]


def filter_by_title(
    memories,
    search_text: str,
):

    if not search_text:
        return list(memories)

    search_text = (
        search_text
        .strip()
        .lower()
    )

    if not search_text:
        return list(memories)

    return [
        memory
        for memory in memories
        if search_text
        in memory["title"].lower()
    ]


def open_memory(
    memory_id: str,
):

    st.session_state.selected_memory_id = (
        memory_id
    )

    st.query_params["memory_id"] = (
        memory_id
    )

    st.switch_page(
        "pages/memory_detail.py"
    )


def clear_memory_selection():

    st.session_state.selected_memory_ids = (
        set()
    )


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

        selected_ids.add(
            memory_id
        )

    else:

        selected_ids.discard(
            memory_id
        )

    st.session_state.selected_memory_ids = (
        selected_ids
    )


# ============================================================
# Compact Dashboard UI
# ============================================================

st.markdown(
    """
    <style>

    [data-testid="stMainBlockContainer"] {
        padding-top: 3.75rem;
    }


    .st-key-dashboard-sticky-header {
        padding-top: 0.20rem;
        padding-bottom: 0.45rem;

        margin-bottom: 0.35rem;

        border-bottom: 1px solid
            rgba(128, 128, 128, 0.18);
    }


    .st-key-dashboard-sticky-header h1 {
        margin-top: 0;
        margin-bottom: 0.10rem;

        padding-top: 0;
        padding-bottom: 0;

        line-height: 1.20;
    }


    .st-key-dashboard-sticky-header
    [data-testid="stMarkdownContainer"] p {
        margin-top: 0.05rem;
        margin-bottom: 0.30rem;
    }


    .st-key-dashboard-sticky-header
    [data-testid="stHorizontalBlock"] {
        gap: 0.60rem;
    }


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


    .st-key-dashboard-sticky-header
    button:disabled {
        opacity: 0.45;
        cursor: not-allowed;
    }


    .st-key-dashboard-sticky-header
    .stTabs [data-baseweb="tab-list"] {
        padding-top: 0.05rem;
        padding-bottom: 0.05rem;

        margin-top: 0.10rem;
    }


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
# ============================================================

# Validate selected IDs only; avoid loading every memory on each rerun.
st.session_state.selected_memory_ids = get_existing_memory_ids(
    USER_ID,
    st.session_state.get("selected_memory_ids", set()),
)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header(
        "Memory Vault"
    )


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


    st.subheader(
        "Quick Search"
    )


    search_text = st.text_input(
        "Search memories...",
        placeholder="Search by title",
        key="memory_title_search",
    )


    st.caption(
        "Title search only — no RAG or AI."
    )


    st.divider()


    st.subheader(
        "Demo"
    )


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
# Dashboard Header
# ============================================================

with st.container(
    key="dashboard_sticky_header"
):

    st.title(
        "🧠 Memory Vault"
    )

    st.write(
        "Your personal AI memory assistant."
    )


    # ========================================================
    # Dashboard Controls
    # ========================================================

    (
        control_col1,
        control_col2,
        control_col3,
        control_col4,
    ) = st.columns(
        [1.4, 1.4, 3.0, 1.6]
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


    # --------------------------------------------------------
    # Ask Question
    # --------------------------------------------------------

    with control_col4:

        if st.button(
            "Ask Question",
            type="primary",
            use_container_width=True,
            key="homepage_ask_question",
        ):

            st.query_params["scope"] = (
                "all"
            )

            st.query_params["thread_id"] = (
                create_chat_thread_id(
                    USER_ID
                )
            )

            st.switch_page(
                "pages/chat.py"
            )


    # ========================================================
    # Counts
    # ========================================================

    memory_counts = count_memories_by_type(USER_ID, search_text)
    all_count = memory_counts["all_count"]
    photo_count = memory_counts["photo_count"]
    document_count = memory_counts["document_count"]
    voice_count = memory_counts["voice_count"]


    # ========================================================
    # Dashboard Tabs
    # ========================================================

    (
        all_tab,
        photo_tab,
        document_tab,
        voice_tab,
    ) = st.tabs(
        [
            f"All ({all_count})",
            f"Photos ({photo_count})",
            f"Documents ({document_count})",
            f"Voice ({voice_count})",
        ]
    )


# ============================================================
# Memory Card Grid
# ============================================================

def render_memory_grid(memory_type, tab_name: str):
    page_key = f"memory_page_{tab_name}"
    search_key = f"{page_key}_search"
    if st.session_state.get(search_key) != search_text:
        st.session_state[page_key] = 1
        st.session_state[search_key] = search_text

    count_key = {
        "All": "all_count", "Photos": "photo_count",
        "Documents": "document_count", "Voice": "voice_count",
    }[tab_name]
    total_count = memory_counts[count_key]
    page_size = 20
    total_pages = max(1, (total_count + page_size - 1) // page_size)
    page = min(max(1, st.session_state.get(page_key, 1)), total_pages)
    st.session_state[page_key] = page
    offset = (page - 1) * page_size

    page_memories = list_memories_page(
        USER_ID, limit=page_size, offset=offset,
        memory_type=memory_type, search_text=search_text,
    )
    if not page_memories:
        if search_text:
            st.info("No memories match your title search.")
        else:
            st.info(
                f"No {tab_name.lower()} memories found. Please add memories "
                "or load demo memories from the sidebar."
            )
        return

    st.caption(
        f"Showing {offset + 1}–{offset + len(page_memories)} of "
        f"{total_count} {tab_name.lower()} memories."
    )
    columns = st.columns(3)
    for index, memory in enumerate(page_memories):
        with columns[index % 3]:
            with st.container(border=True):
                icon = get_type_icon(memory["memory_type"])
                type_label = get_type_label(memory["memory_type"])
                st.markdown(f"### {icon} {type_label}")
                st.write(f"**{memory['title']}**")
                st.caption(format_date(memory["created_at"]))
                preview = get_summary_preview(memory)
                if preview:
                    st.write(preview)
                else:
                    st.caption("Open memory to generate summary")

                if st.session_state.get("edit_memories_mode", False):
                    checkbox_key = f"select_{tab_name}_{memory['id']}"
                    if checkbox_key not in st.session_state:
                        st.session_state[checkbox_key] = (
                            memory["id"] in st.session_state.get(
                                "selected_memory_ids", set()
                            )
                        )
                    st.checkbox(
                        "Select", key=checkbox_key,
                        on_change=sync_memory_selection,
                        args=(memory["id"], checkbox_key),
                    )

                if st.button(
                    "Open Memory →", key=f"open_{tab_name}_{memory['id']}",
                    use_container_width=True,
                ):
                    open_memory(memory["id"])

    if total_pages > 1:
        previous_col, page_col, next_col = st.columns([1, 2, 1])
        with previous_col:
            if st.button(
                "← Previous", key=f"{page_key}_previous",
                disabled=page <= 1, use_container_width=True,
            ):
                st.session_state[page_key] = page - 1
                st.rerun()
        with page_col:
            st.caption(f"Page {page} of {total_pages}")
        with next_col:
            if st.button(
                "Next →", key=f"{page_key}_next",
                disabled=page >= total_pages, use_container_width=True,
            ):
                st.session_state[page_key] = page + 1
                st.rerun()


# ============================================================
# Dashboard Tabs Content
# ============================================================

with all_tab:
    render_memory_grid(None, "All")


with photo_tab:
    render_memory_grid("image", "Photos")


with document_tab:
    render_memory_grid("document", "Documents")


with voice_tab:
    render_memory_grid("audio", "Voice")
