import streamlit as st

from agent.graph import (
    create_thread_id,
    get_conversation_history,
    get_thread_state,
    run_langgraph_agent,
    _checkpointer,
)

from config.settings import settings
from database.database import initialize_database

from memory.memory_service import (
    get_memory_by_id,
    list_memories,
)

from rag.retriever import retrieve
from rag.grounding import (
    DEFAULT_DISTANCE_THRESHOLD,
    get_grounded_results,
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Memory Vault Chat",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# Database
# ============================================================

initialize_database()


# ============================================================
# Current User
# ============================================================

USER_ID = settings.DEMO_USER_ID


# ============================================================
# Constants
# ============================================================

# These are engineering/UI signals.
# They are NOT probabilities.

HIGH_RETRIEVAL_DISTANCE = 0.4

MEDIUM_RETRIEVAL_DISTANCE = (
    DEFAULT_DISTANCE_THRESHOLD
)

NOT_FOUND_MESSAGE = (
    "This information wasn't found in your memory."
)


# ============================================================
# Safe Memory Lookup
# ============================================================

def get_existing_memory(
    memory_id: str | None,
):
    """
    Return a user-owned memory if it still exists.

    If the memory was deleted, return None instead
    of allowing the chat page to crash.
    """

    if not memory_id:
        return None

    try:
        return get_memory_by_id(
            memory_id=memory_id,
            user_id=USER_ID,
        )

    except ValueError:
        return None


# ============================================================
# Scope Helpers
# ============================================================

def normalize_scope(
    value: str | None,
) -> str:
    """
    Normalize chat scope.

    Only two scopes are supported:
        all
        memory
    """

    if value == "memory":
        return "memory"

    return "all"


def get_scope_memory_id():
    """
    Return the current memory ID when the chat is
    memory-scoped.

    Return None for global scope.
    """

    if (
        st.session_state.get("chat_scope")
        == "memory"
    ):
        return st.session_state.get(
            "chat_memory_id"
        )

    return None


# ============================================================
# Message Helpers
# ============================================================

def get_message_type(
    message,
) -> str:
    return getattr(
        message,
        "type",
        "",
    )


def get_message_content(
    message,
) -> str:

    content = getattr(
        message,
        "content",
        "",
    )

    if isinstance(
        content,
        str,
    ):
        return content

    return str(content)


def get_previous_human_question(
    messages,
    current_index: int,
):
    """
    Find the human question immediately associated
    with an assistant response.
    """

    for index in range(
        current_index - 1,
        -1,
        -1,
    ):

        if (
            get_message_type(
                messages[index]
            )
            == "human"
        ):

            return get_message_content(
                messages[index]
            )

    return None


# ============================================================
# Retrieval Cache
# ============================================================

def get_cache_key(
    thread_id: str,
    question: str,
    memory_id: str | None,
):
    """
    Build a unique retrieval-cache key.

    Thread ID is intentionally included so that the
    same question in two different conversations does
    not accidentally share UI evidence.
    """

    scope_key = (
        memory_id
        if memory_id
        else "all"
    )

    return (
        f"{thread_id}|"
        f"{scope_key}|"
        f"{question.strip()}"
    )


def get_empty_retrieval_signal():
    return {
        "grounded": False,
        "confidence": "Not found",
        "sources": [],
    }


def get_retrieval_signal(
    question: str,
    memory_id: str | None,
):
    """
    Perform retrieval once and convert it into the
    confidence/source information used by the UI.

    IMPORTANT:
    This function should only be called when the
    retrieval signal is not already cached.
    """

    results = retrieve(
        question=question,
        user_id=USER_ID,
        top_k=3,
        memory_id=memory_id,
    )

    grounded_results = get_grounded_results(
        results
    )

    if not grounded_results:
        return get_empty_retrieval_signal()

    # --------------------------------------------------------
    # Strongest match
    # --------------------------------------------------------

    strongest_distance = min(
        distance
        for _, distance in grounded_results
    )

    if (
        strongest_distance
        <= HIGH_RETRIEVAL_DISTANCE
    ):

        confidence = (
            "High retrieval confidence"
        )

    elif (
        strongest_distance
        <= MEDIUM_RETRIEVAL_DISTANCE
    ):

        confidence = (
            "Medium retrieval confidence"
        )

    else:

        confidence = (
            "Low retrieval confidence"
        )

    # --------------------------------------------------------
    # Build memory lookup
    # --------------------------------------------------------

    memories = list_memories(
        USER_ID
    )

    memory_map = {
        memory["id"]: memory
        for memory in memories
    }

    # --------------------------------------------------------
    # Build source information
    # --------------------------------------------------------

    sources = []

    for (
        document,
        distance,
    ) in grounded_results:

        source_memory_id = (
            document.metadata.get(
                "memory_id"
            )
        )

        source_memory = memory_map.get(
            source_memory_id
        )

        sources.append(
            {
                "memory_id":
                    source_memory_id,

                "title": (
                    source_memory["title"]
                    if source_memory is not None
                    else source_memory_id
                ),

                "memory_type":
                    document.metadata.get(
                        "memory_type",
                        "unknown",
                    ),

                "chunk_index":
                    document.metadata.get(
                        "chunk_index",
                        0,
                    ),

                "distance":
                    distance,
            }
        )

    return {
        "grounded": True,
        "confidence": confidence,
        "sources": sources,
    }


def get_or_create_retrieval_signal(
    thread_id: str,
    question: str,
    memory_id: str | None,
):
    """
    Return cached retrieval information.

    If it does not exist, calculate it once.

    This is the important fix for conversation history:
    rendering old messages does not repeatedly call Chroma
    after the first retrieval for that message.
    """

    cache = st.session_state[
        "chat_retrieval_cache"
    ]

    cache_key = get_cache_key(
        thread_id=thread_id,
        question=question,
        memory_id=memory_id,
    )

    if cache_key in cache:
        return cache[cache_key]

    try:

        signal = get_retrieval_signal(
            question=question,
            memory_id=memory_id,
        )

    except Exception:

        signal = (
            get_empty_retrieval_signal()
        )

    cache[cache_key] = signal

    return signal


# ============================================================
# Navigation
# ============================================================

def go_to_dashboard():
    """
    Return to the main dashboard.
    """

    st.query_params.clear()

    st.switch_page(
        "app.py"
    )


# ============================================================
# New Conversation
# ============================================================

def switch_to_new_conversation():
    """
    Create a completely new LangGraph thread.

    The current scope is preserved.

    Previous conversations remain persisted in the
    checkpointer and therefore remain available in
    Chat History.
    """

    new_thread_id = create_thread_id(
        USER_ID
    )

    current_scope = (
        st.session_state.get(
            "chat_scope",
            "all",
        )
    )

    current_memory_id = (
        st.session_state.get(
            "chat_memory_id"
        )
    )

    # --------------------------------------------------------
    # Update current session
    # --------------------------------------------------------

    st.session_state.chat_thread_id = (
        new_thread_id
    )

    # IMPORTANT:
    # Do NOT clear the complete retrieval cache.
    #
    # Old conversation signals are keyed by thread ID,
    # so they cannot collide with the new conversation.
    #
    # This also means opening the old conversation later
    # does not require another retrieval.

    st.query_params["thread_id"] = (
        new_thread_id
    )

    # --------------------------------------------------------
    # Preserve scope
    # --------------------------------------------------------

    if (
        current_scope == "memory"
        and current_memory_id
        and get_existing_memory(
            current_memory_id
        ) is not None
    ):

        st.session_state.chat_scope = (
            "memory"
        )

        st.session_state.chat_memory_id = (
            current_memory_id
        )

        st.query_params["scope"] = (
            "memory"
        )

        st.query_params["memory_id"] = (
            current_memory_id
        )

    else:

        st.session_state.chat_scope = (
            "all"
        )

        st.session_state.chat_memory_id = (
            None
        )

        st.query_params["scope"] = (
            "all"
        )

        if (
            "memory_id"
            in st.query_params
        ):

            del st.query_params[
                "memory_id"
            ]

    # --------------------------------------------------------
    # Streamlit must rerun to render the new thread.
    # This is a UI rerun, NOT an LLM rerun.
    # --------------------------------------------------------

    st.rerun()


# ============================================================
# Delete Conversation
# ============================================================

def delete_conversation(
    thread_id: str,
):
    """
    Delete one persisted LangGraph conversation.

    Only the current user's thread IDs are allowed.
    """

    if (
        not thread_id
        or not thread_id.strip()
    ):

        raise ValueError(
            "Thread ID is required."
        )

    user_prefix = (
        f"{USER_ID}:"
    )

    if not thread_id.startswith(
        user_prefix
    ):

        raise ValueError(
            "You are not allowed to delete this conversation."
        )

    _checkpointer.delete_thread(
        thread_id
    )


# ============================================================
# Open Conversation
# ============================================================

def open_conversation(
    conversation,
):
    """
    Load an existing conversation.

    IMPORTANT:
    We intentionally DO NOT clear the retrieval cache.

    This prevents historical questions from being
    re-retrieved unnecessarily after loading a conversation.
    """

    conversation_thread_id = (
        conversation["thread_id"]
    )

    # --------------------------------------------------------
    # Security validation
    # --------------------------------------------------------

    user_prefix = (
        f"{USER_ID}:"
    )

    if not conversation_thread_id.startswith(
        user_prefix
    ):

        st.error(
            "This conversation does not belong to the current user."
        )

        return

    # --------------------------------------------------------
    # Restore thread
    # --------------------------------------------------------

    st.session_state.chat_thread_id = (
        conversation_thread_id
    )

    conversation_scope = normalize_scope(
        conversation.get(
            "scope",
            "all",
        )
    )

    conversation_memory_id = (
        conversation.get(
            "memory_id"
        )
    )

    # --------------------------------------------------------
    # Restore memory scope
    # --------------------------------------------------------

    if (
        conversation_scope == "memory"
        and conversation_memory_id
    ):

        memory = get_existing_memory(
            conversation_memory_id
        )

        if memory is not None:

            st.session_state.chat_scope = (
                "memory"
            )

            st.session_state.chat_memory_id = (
                conversation_memory_id
            )

            st.query_params["scope"] = (
                "memory"
            )

            st.query_params[
                "memory_id"
            ] = conversation_memory_id

        else:

            # ------------------------------------------------
            # Memory was deleted after this conversation
            # was created.
            #
            # Safely fall back to global scope.
            # ------------------------------------------------

            st.session_state.chat_scope = (
                "all"
            )

            st.session_state.chat_memory_id = (
                None
            )

            st.query_params["scope"] = (
                "all"
            )

            if (
                "memory_id"
                in st.query_params
            ):

                del st.query_params[
                    "memory_id"
                ]

    else:

        st.session_state.chat_scope = (
            "all"
        )

        st.session_state.chat_memory_id = (
            None
        )

        st.query_params["scope"] = (
            "all"
        )

        if (
            "memory_id"
            in st.query_params
        ):

            del st.query_params[
                "memory_id"
            ]

    # --------------------------------------------------------
    # Restore thread URL
    # --------------------------------------------------------

    st.query_params["thread_id"] = (
        conversation_thread_id
    )

    # --------------------------------------------------------
    # UI rerun only.
    # --------------------------------------------------------

    st.rerun()


# ============================================================
# Resolve Thread
# ============================================================

query_thread_id = (
    st.query_params.get(
        "thread_id"
    )
)

if query_thread_id:

    thread_id = (
        query_thread_id
    )

else:

    thread_id = create_thread_id(
        USER_ID
    )

    st.query_params["thread_id"] = (
        thread_id
    )


# ============================================================
# Validate Thread Ownership
# ============================================================

if not thread_id.startswith(
    f"{USER_ID}:"
):

    st.error(
        "Invalid conversation thread."
    )

    if st.button(
        "← Back to Dashboard",
        type="primary",
    ):

        go_to_dashboard()

    st.stop()


# ============================================================
# Resolve Initial Scope
# ============================================================

query_scope = normalize_scope(
    st.query_params.get(
        "scope"
    )
)

query_memory_id = (
    st.query_params.get(
        "memory_id"
    )
)


# ============================================================
# Validate Initial Memory Scope
# ============================================================

if query_scope == "memory":

    if not query_memory_id:

        st.error(
            "No memory was selected for this chat."
        )

        if st.button(
            "← Back to Dashboard",
            type="primary",
        ):

            go_to_dashboard()

        st.stop()

    selected_memory = (
        get_existing_memory(
            query_memory_id
        )
    )

    if selected_memory is None:

        st.error(
            "The selected memory could not be found."
        )

        if st.button(
            "← Back to Dashboard",
            type="primary",
        ):

            go_to_dashboard()

        st.stop()

else:

    selected_memory = None
    query_memory_id = None


# ============================================================
# Session State
# ============================================================

if (
    "chat_retrieval_cache"
    not in st.session_state
):

    st.session_state.chat_retrieval_cache = {}


# ------------------------------------------------------------
# Thread state
# ------------------------------------------------------------

if (
    "chat_thread_id"
    not in st.session_state
    or
    st.session_state.chat_thread_id
    != thread_id
):

    st.session_state.chat_thread_id = (
        thread_id
    )

    st.session_state.chat_scope = (
        query_scope
    )

    st.session_state.chat_memory_id = (
        query_memory_id
    )


# ------------------------------------------------------------
# Scope defaults
# ------------------------------------------------------------

if (
    "chat_scope"
    not in st.session_state
):

    st.session_state.chat_scope = (
        query_scope
    )


if (
    "chat_memory_id"
    not in st.session_state
):

    st.session_state.chat_memory_id = (
        query_memory_id
    )


# ============================================================
# Validate Current Session Memory
# ============================================================

if (
    st.session_state.chat_scope
    == "memory"
):

    current_memory_id = (
        st.session_state.chat_memory_id
    )

    current_memory = (
        get_existing_memory(
            current_memory_id
        )
    )

    if current_memory is None:

        st.session_state.chat_scope = (
            "all"
        )

        st.session_state.chat_memory_id = (
            None
        )

        st.query_params["scope"] = (
            "all"
        )

        if (
            "memory_id"
            in st.query_params
        ):

            del st.query_params[
                "memory_id"
            ]


# ============================================================
# Header
# ============================================================

if st.button(
    "← Back to Dashboard",
    key="chat_back_dashboard",
):

    go_to_dashboard()


st.title(
    "🧠 Memory Vault Chat"
)

st.caption(
    "Conversational questions over your saved memories."
)


# ============================================================
# New Conversation + Chat History
# ============================================================

st.divider()

button_col1, button_col2 = st.columns(
    [1, 1]
)


# ============================================================
# New Conversation
# ============================================================

with button_col1:

    if st.button(
        "＋ New Conversation",
        use_container_width=True,
        key="chat_new_conversation",
    ):

        switch_to_new_conversation()


# ============================================================
# Chat History
# ============================================================

with button_col2:

    with st.popover(
        "💬 Chat History",
        use_container_width=True,
    ):

        st.caption(
            "Your saved conversations"
        )

        st.divider()

        # ----------------------------------------------------
        # Load history
        # ----------------------------------------------------

        try:

            conversations = list(
                get_conversation_history(
                    USER_ID
                )
            )

            # Newest → oldest
            conversations = list(
                reversed(
                    conversations
                )
            )

        except Exception as error:

            conversations = []

            st.warning(
                "Conversation history could not be loaded. "
                f"{error}"
            )

        # ----------------------------------------------------
        # No history
        # ----------------------------------------------------

        if not conversations:

            st.caption(
                "No previous conversations yet."
            )

        else:

            current_thread_id = (
                st.session_state.chat_thread_id
            )

            # ------------------------------------------------
            # Chat History CSS
            # ------------------------------------------------

            st.markdown(
                """
                <style>

                /* -------------------------------------------
                   History title buttons
                   ------------------------------------------- */

                [data-testid="stPopoverBody"] button {
                    justify-content: flex-start !important;
                    text-align: left !important;
                }

                [data-testid="stPopoverBody"] button > div {
                    width: 100% !important;
                    justify-content: flex-start !important;
                    text-align: left !important;
                }

                [data-testid="stPopoverBody"]
                button
                div[data-testid="stMarkdownContainer"] {
                    width: 100% !important;
                    flex: 1 1 auto !important;
                    justify-content: flex-start !important;
                    text-align: left !important;
                }

                [data-testid="stPopoverBody"]
                button
                div[data-testid="stMarkdownContainer"] p {
                    width: 100% !important;
                    margin: 0 !important;
                    text-align: left !important;
                }


                /* -------------------------------------------
                   Delete button
                   ------------------------------------------- */

                [data-testid="stPopoverBody"]
                div[class*="st-key-delete_history_"]
                button {

                    width: 2.2rem !important;
                    min-width: 2.2rem !important;
                    max-width: 2.2rem !important;

                    height: 2.2rem !important;
                    min-height: 2.2rem !important;

                    padding: 0 !important;
                    margin: 0 !important;
                }

                [data-testid="stPopoverBody"]
                div[class*="st-key-delete_history_"]
                button > div {

                    width: 100% !important;
                    justify-content: center !important;
                    text-align: center !important;

                    padding: 0 !important;
                }

                [data-testid="stPopoverBody"]
                div[class*="st-key-delete_history_"]
                div[data-testid="stMarkdownContainer"] {

                    width: 100% !important;
                    flex: 0 0 auto !important;

                    justify-content: center !important;
                    text-align: center !important;
                }

                [data-testid="stPopoverBody"]
                div[class*="st-key-delete_history_"]
                div[data-testid="stMarkdownContainer"] p {

                    width: 100% !important;
                    margin: 0 !important;

                    text-align: center !important;
                }

                </style>
                """,
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # Render newest → oldest
            # ------------------------------------------------

            for conversation in conversations:

                conversation_thread_id = (
                    conversation["thread_id"]
                )

                # ------------------------------------------------
                # Safety: ignore threads belonging to another user
                # ------------------------------------------------

                if not conversation_thread_id.startswith(
                    f"{USER_ID}:"
                ):

                    continue

                title = conversation.get(
                    "title",
                    "New Conversation",
                )

                if not title:

                    title = (
                        "New Conversation"
                    )

                title = str(title)

                # ------------------------------------------------
                # Compact title
                # ------------------------------------------------

                if len(title) > 42:

                    display_title = (
                        title[:42]
                        + "..."
                    )

                else:

                    display_title = title

                # ------------------------------------------------
                # Current conversation marker
                # ------------------------------------------------

                if (
                    conversation_thread_id
                    == current_thread_id
                ):

                    prefix = "●"

                else:

                    prefix = "○"

                button_label = (
                    f"{prefix} {display_title}"
                )

                # ------------------------------------------------
                # History item layout
                # ------------------------------------------------

                history_col, delete_col = (
                    st.columns(
                        [0.86, 0.14]
                    )
                )

                with history_col:

                    if st.button(
                        button_label,
                        use_container_width=True,
                        key=(
                            "history_"
                            + conversation_thread_id
                        ),
                    ):

                        open_conversation(
                            conversation
                        )

                with delete_col:

                    if st.button(
                        "🗑️",
                        help="Delete conversation",
                        key=(
                            "delete_history_"
                            + conversation_thread_id
                        ),
                    ):

                        try:

                            delete_conversation(
                                conversation_thread_id
                            )

                            # ------------------------------------------------
                            # If current conversation was deleted,
                            # create a fresh thread.
                            # ------------------------------------------------

                            if (
                                conversation_thread_id
                                == current_thread_id
                            ):

                                new_thread_id = (
                                    create_thread_id(
                                        USER_ID
                                    )
                                )

                                st.session_state.chat_thread_id = (
                                    new_thread_id
                                )

                                st.session_state.chat_scope = (
                                    "all"
                                )

                                st.session_state.chat_memory_id = (
                                    None
                                )

                                st.query_params[
                                    "thread_id"
                                ] = new_thread_id

                                st.query_params[
                                    "scope"
                                ] = "all"

                                if (
                                    "memory_id"
                                    in st.query_params
                                ):

                                    del st.query_params[
                                        "memory_id"
                                    ]

                            st.toast(
                                "Conversation deleted.",
                                icon="🗑️",
                            )

                            st.rerun()

                        except Exception as error:

                            st.error(
                                "Could not delete the conversation. "
                                f"{error}"
                            )


# ============================================================
# Scope Selector
# ============================================================

st.divider()

st.subheader(
    "Chat Scope"
)


scope_options = [
    "All Memories",
    "This Memory",
]


current_scope_label = (
    "This Memory"
    if (
        st.session_state.chat_scope
        == "memory"
    )
    else "All Memories"
)


selected_scope_label = st.radio(
    "Scope",
    options=scope_options,
    index=scope_options.index(
        current_scope_label
    ),
    horizontal=True,
    key="chat_scope_selector",
)


new_scope = (
    "memory"
    if selected_scope_label
    == "This Memory"
    else "all"
)


# ============================================================
# Specific Memory Selector
# ============================================================

if new_scope == "memory":

    memories = list_memories(
        USER_ID
    )

    if not memories:

        st.info(
            "You don't have any memories to select."
        )

        st.session_state.chat_scope = (
            "all"
        )

        st.session_state.chat_memory_id = (
            None
        )

        st.query_params["scope"] = (
            "all"
        )

        if (
            "memory_id"
            in st.query_params
        ):

            del st.query_params[
                "memory_id"
            ]

    else:

        # ----------------------------------------------------
        # Avoid duplicate titles breaking the dictionary.
        # ----------------------------------------------------

        memory_options = {}

        for memory in memories:

            title = (
                memory["title"]
                or "Untitled Memory"
            )

            memory_id = memory["id"]

            display_title = title

            # If duplicate titles exist, make the
            # select-box options unique.
            if display_title in memory_options:

                display_title = (
                    f"{title} "
                    f"({memory_id[:8]})"
                )

            memory_options[
                display_title
            ] = memory_id

        option_titles = list(
            memory_options.keys()
        )

        current_memory_id = (
            st.session_state.chat_memory_id
        )

        default_index = 0

        for index, title in enumerate(
            option_titles
        ):

            if (
                memory_options[title]
                == current_memory_id
            ):

                default_index = index
                break

        selected_title = st.selectbox(
            "Select memory",
            options=option_titles,
            index=default_index,
            key="chat_memory_selector",
        )

        selected_memory_id = (
            memory_options[
                selected_title
            ]
        )

        st.session_state.chat_scope = (
            "memory"
        )

        st.session_state.chat_memory_id = (
            selected_memory_id
        )

        st.query_params["scope"] = (
            "memory"
        )

        st.query_params["memory_id"] = (
            selected_memory_id
        )


else:

    st.session_state.chat_scope = (
        "all"
    )

    st.session_state.chat_memory_id = (
        None
    )

    st.query_params["scope"] = (
        "all"
    )

    if (
        "memory_id"
        in st.query_params
    ):

        del st.query_params[
            "memory_id"
        ]


# ============================================================
# Current Scope Indicator
# ============================================================

if (
    st.session_state.chat_scope
    == "memory"
):

    scope_memory = (
        get_existing_memory(
            st.session_state.chat_memory_id
        )
    )

    if scope_memory is not None:

        st.info(
            "Current scope: "
            f"**{scope_memory['title']}**"
        )

    else:

        # Safety fallback if the memory was deleted
        # while the page was open.

        st.session_state.chat_scope = (
            "all"
        )

        st.session_state.chat_memory_id = (
            None
        )

        st.info(
            "Current scope: **All Memories**"
        )

else:

    st.info(
        "Current scope: **All Memories**"
    )


st.divider()


# ============================================================
# Load Persistent Conversation
# ============================================================

try:

    thread_state = get_thread_state(
        user_id=USER_ID,
        thread_id=thread_id,
    )

    conversation_messages = (
        thread_state.values.get(
            "messages",
            [],
        )
    )

except Exception as error:

    conversation_messages = []

    st.warning(
        "The conversation could not be loaded. "
        f"{error}"
    )


# ============================================================
# Conversation Messages
# ============================================================

for index, message in enumerate(
    conversation_messages
):

    message_type = (
        get_message_type(
            message
        )
    )

    # ========================================================
    # User Message
    # ========================================================

    if message_type == "human":

        with st.chat_message(
            "user"
        ):

            st.write(
                get_message_content(
                    message
                )
            )


    # ========================================================
    # Assistant Message
    # ========================================================

    elif message_type == "ai":

        content = (
            get_message_content(
                message
            )
        )

        # Tool-calling intermediate AI messages can
        # have empty user-facing content.

        if not content.strip():

            continue

        with st.chat_message(
            "assistant"
        ):

            st.write(
                content
            )

            # ------------------------------------------------
            # Find associated question
            # ------------------------------------------------

            question = (
                get_previous_human_question(
                    conversation_messages,
                    index,
                )
            )

            if not question:

                continue

            # ------------------------------------------------
            # IMPORTANT FIX
            #
            # Use the scope that belongs to this
            # conversation, not a temporary UI selection.
            #
            # For an old conversation, its stored scope
            # should be used whenever possible.
            # ------------------------------------------------

            conversation_scope = normalize_scope(
                st.session_state.get(
                    "chat_scope",
                    "all",
                )
            )

            conversation_memory_id = (
                st.session_state.get(
                    "chat_memory_id"
                )
                if conversation_scope
                == "memory"
                else None
            )

            # ------------------------------------------------
            # Retrieval signal
            #
            # This uses the cache.
            #
            # Therefore loading history does NOT cause
            # repeated retrieval for the same question
            # once it has already been calculated.
            # ------------------------------------------------

            signal = (
                get_or_create_retrieval_signal(
                    thread_id=thread_id,
                    question=question,
                    memory_id=conversation_memory_id,
                )
            )

            # ------------------------------------------------
            # Grounded answer
            # ------------------------------------------------

            if signal["grounded"]:

                st.caption(
                    "Confidence: "
                    f"{signal['confidence']}"
                )

                # ------------------------------------------------
                # Sources
                # ------------------------------------------------

                if signal["sources"]:

                    with st.expander(
                        "Sources"
                    ):

                        for source in (
                            signal["sources"]
                        ):

                            st.write(
                                (
                                    f"**{source['title']}** "
                                    f"• chunk "
                                    f"{source['chunk_index']} "
                                    f"• distance "
                                    f"{source['distance']:.4f}"
                                )
                            )

            # ------------------------------------------------
            # Not found
            # ------------------------------------------------

            else:

                if (
                    NOT_FOUND_MESSAGE
                    in content
                ):

                    st.caption(
                        "Not found"
                    )


# ============================================================
# Chat Input
# ============================================================

question = st.chat_input(
    "Ask a follow-up question..."
)


# ============================================================
# Process New Question
# ============================================================

if question:

    question = question.strip()

    if not question:

        st.toast(
            "Please enter a question.",
            icon="⚠️",
        )

    else:

        current_memory_id = (
            get_scope_memory_id()
        )

        # ----------------------------------------------------
        # Validate selected memory immediately before
        # running the agent.
        # ----------------------------------------------------

        if current_memory_id:

            if (
                get_existing_memory(
                    current_memory_id
                )
                is None
            ):

                st.session_state.chat_scope = (
                    "all"
                )

                st.session_state.chat_memory_id = (
                    None
                )

                st.query_params["scope"] = (
                    "all"
                )

                if (
                    "memory_id"
                    in st.query_params
                ):

                    del st.query_params[
                        "memory_id"
                    ]

                st.toast(
                    "The selected memory no longer exists.",
                    icon="⚠️",
                )

                st.rerun()

        try:

            with st.spinner(
                "Memory Vault is thinking..."
            ):

                run_langgraph_agent(
                    question=question,
                    user_id=USER_ID,
                    thread_id=thread_id,
                    memory_id=current_memory_id,
                )

            # ------------------------------------------------
            # Calculate retrieval evidence ONCE.
            #
            # The result is stored in the cache.
            #
            # When Streamlit reruns and renders this new
            # conversation, the retrieval will NOT happen
            # again.
            # ------------------------------------------------

            get_or_create_retrieval_signal(
                thread_id=thread_id,
                question=question,
                memory_id=current_memory_id,
            )

            # ------------------------------------------------
            # UI rerun.
            #
            # This rerun only renders the newly persisted
            # LangGraph messages.
            #
            # It does NOT call run_langgraph_agent again.
            # It does NOT call retrieval again because
            # the signal is already cached.
            # ------------------------------------------------

            st.rerun()

        except Exception as error:

            st.toast(
                (
                    "Something went wrong while "
                    f"running the chat: {error}"
                ),
                icon="❌",
            )