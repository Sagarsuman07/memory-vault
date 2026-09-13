import streamlit as st

from agent.graph import (
    create_thread_id,
    get_conversation_history,
    get_thread_state,
    run_langgraph_agent,
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


initialize_database()


USER_ID = settings.DEMO_USER_ID


# ============================================================
# Constants
# ============================================================

# Engineering/UI signal only.
# These are not probabilities.

HIGH_RETRIEVAL_DISTANCE = 0.4

MEDIUM_RETRIEVAL_DISTANCE = (
    DEFAULT_DISTANCE_THRESHOLD
)

NOT_FOUND_MESSAGE = (
    "This information wasn't found in your memory."
)


# ============================================================
# Helpers
# ============================================================

def normalize_scope(
    value: str | None,
) -> str:

    if value == "memory":
        return "memory"

    return "all"


# ============================================================
# Retrieval Confidence
# ============================================================

def get_retrieval_signal(
    question: str,
    memory_id: str | None,
):

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

        return {
            "grounded": False,
            "confidence": "Not found",
            "sources": [],
        }

    # ========================================================
    # Strongest relevant match = lowest distance.
    # ========================================================

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

    # ========================================================
    # Build source information
    # ========================================================

    sources = []

    memory_map = {
        memory["id"]: memory
        for memory in list_memories(
            USER_ID
        )
    }

    for document, distance in grounded_results:

        source_memory_id = (
            document.metadata[
                "memory_id"
            ]
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
                    document.metadata[
                        "memory_type"
                    ],

                "chunk_index":
                    document.metadata[
                        "chunk_index"
                    ],

                "distance":
                    distance,
            }
        )

    return {
        "grounded": True,
        "confidence": confidence,
        "sources": sources,
    }


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

    return str(
        content
    )


def get_previous_human_question(
    messages,
    current_index: int,
):

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
# Cache Key
# ============================================================

def get_cache_key(
    thread_id: str,
    question: str,
    memory_id: str | None,
):

    scope_key = (
        memory_id
        or "all"
    )

    return (
        f"{thread_id}|"
        f"{scope_key}|"
        f"{question}"
    )


# ============================================================
# Navigation
# ============================================================

def go_to_dashboard():

    st.query_params.clear()

    st.switch_page(
        "app.py"
    )


def switch_to_new_conversation():
    """
    Shared logic for starting a new thread while
    preserving the current chat scope.
    """

    new_thread_id = create_thread_id(
        USER_ID
    )

    current_scope = (
        st.session_state.chat_scope
    )

    current_memory_id = (
        st.session_state.chat_memory_id
    )

    st.session_state.chat_thread_id = (
        new_thread_id
    )

    st.session_state.chat_retrieval_cache = {}

    st.query_params["thread_id"] = (
        new_thread_id
    )

    if current_scope == "memory":

        st.query_params["scope"] = (
            "memory"
        )

        if current_memory_id:

            st.query_params["memory_id"] = (
                current_memory_id
            )

    else:

        st.query_params["scope"] = (
            "all"
        )

        if "memory_id" in st.query_params:

            del st.query_params[
                "memory_id"
            ]

    st.rerun()


def open_conversation(
    conversation,
):
    """
    Switch to an existing conversation and
    restore its saved scope.
    """

    conversation_thread_id = (
        conversation["thread_id"]
    )

    st.session_state.chat_thread_id = (
        conversation_thread_id
    )

    st.session_state.chat_retrieval_cache = {}

    conversation_scope = (
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
    # Restore scope.
    # --------------------------------------------------------

    if (
        conversation_scope
        == "memory"
        and
        conversation_memory_id
    ):

        memory = get_memory_by_id(
            memory_id=(
                conversation_memory_id
            ),
            user_id=USER_ID,
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

            st.query_params["memory_id"] = (
                conversation_memory_id
            )

        else:

            # ------------------------------------------------
            # Selected memory was deleted.
            # Fall back to global scope.
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

            if "memory_id" in st.query_params:

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

        if "memory_id" in st.query_params:

            del st.query_params[
                "memory_id"
            ]

    st.query_params["thread_id"] = (
        conversation_thread_id
    )

    st.rerun()


# ============================================================
# Resolve Thread
# ============================================================

query_thread_id = st.query_params.get(
    "thread_id"
)


if query_thread_id:

    thread_id = query_thread_id

else:

    thread_id = create_thread_id(
        USER_ID
    )

    st.query_params["thread_id"] = (
        thread_id
    )


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

    selected_memory = get_memory_by_id(
        memory_id=query_memory_id,
        user_id=USER_ID,
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


if "chat_scope" not in st.session_state:

    st.session_state.chat_scope = (
        query_scope
    )


if "chat_memory_id" not in st.session_state:

    st.session_state.chat_memory_id = (
        query_memory_id
    )


# ============================================================
# Validate Current Session Memory
# ============================================================

if (
    st.session_state.chat_scope
    == "memory"
    and
    st.session_state.chat_memory_id
):

    scope_memory = get_memory_by_id(
        memory_id=(
            st.session_state
            .chat_memory_id
        ),
        user_id=USER_ID,
    )

    if scope_memory is None:

        st.session_state.chat_scope = (
            "all"
        )

        st.session_state.chat_memory_id = (
            None
        )


# ============================================================
# Retrieval Cache
# ============================================================

if (
    "chat_retrieval_cache"
    not in st.session_state
):

    st.session_state.chat_retrieval_cache = {}


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
# New Conversation Button
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
        # Load conversation history.
        #
        # IMPORTANT:
        # We explicitly reverse the returned list here.
        # This guarantees that the newest conversation is
        # rendered first and the oldest conversation last.
        # ----------------------------------------------------

        try:

            conversations = list(
                get_conversation_history(
                    USER_ID
                )
            )

            conversations = list(
                reversed(
                    conversations
                )
            )

        except Exception:

            conversations = []

            st.warning(
                "Conversation history could not be loaded."
            )

        # ----------------------------------------------------
        # No conversations
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
            # Left-align history titles.
            #
            # Streamlit renders st.button() using an internal
            # flex container. Target the actual popover body
            # and all nested button elements so the title is
            # aligned from the left edge.
            #
            # This affects ONLY buttons inside Chat History.
            # ------------------------------------------------

            st.markdown(
                """
                <style>

                /* Chat History: left-align the complete button */
                [data-testid="stPopoverBody"] button {
                    justify-content: flex-start !important;
                    text-align: left !important;
                }

                /* Chat History: make the button's inner wrapper
                   use the full available width */
                [data-testid="stPopoverBody"] button > div {
                    width: 100% !important;
                    justify-content: flex-start !important;
                    text-align: left !important;
                }

                /* Chat History: align Streamlit's markdown
                   container to the left */
                [data-testid="stPopoverBody"] button
                div[data-testid="stMarkdownContainer"] {
                    width: 100% !important;
                    flex: 1 1 auto !important;
                    justify-content: flex-start !important;
                    text-align: left !important;
                }

                /* Chat History: align the actual title text */
                [data-testid="stPopoverBody"] button
                div[data-testid="stMarkdownContainer"] p {
                    width: 100% !important;
                    margin: 0 !important;
                    text-align: left !important;
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

                title = (
                    conversation.get(
                        "title",
                        "New Conversation",
                    )
                )

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
                # History button
                #
                # The key starts with "history_" so the CSS
                # selector above can target only these buttons.
                # ------------------------------------------------

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


st.divider()


# ============================================================
# Scope Selector
# ============================================================

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

    else:

        memory_options = {
            memory["title"]:
                memory["id"]
            for memory in memories
        }

        current_memory_id = (
            st.session_state
            .chat_memory_id
        )

        option_titles = list(
            memory_options.keys()
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

    if "memory_id" in st.query_params:

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

    scope_memory = get_memory_by_id(
        memory_id=(
            st.session_state
            .chat_memory_id
        ),
        user_id=USER_ID,
    )

    if scope_memory is not None:

        st.info(
            "Current scope: "
            f"**{scope_memory['title']}**"
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
        "The conversation could not be loaded yet. "
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

        content = get_message_content(
            message
        )

        # ----------------------------------------------------
        # Tool-calling AI messages can have no user-facing
        # content.
        # ----------------------------------------------------

        if not content.strip():

            continue

        with st.chat_message(
            "assistant"
        ):

            st.write(
                content
            )

            # =================================================
            # Find the question associated with this answer.
            # =================================================

            question = (
                get_previous_human_question(
                    conversation_messages,
                    index,
                )
            )

            if question:

                cache_key = (
                    get_cache_key(
                        thread_id,
                        question,
                        (
                            st.session_state
                            .chat_memory_id
                        ),
                    )
                )

                # =================================================
                # Retrieve confidence only once per question.
                # =================================================

                if cache_key not in (
                    st.session_state
                    .chat_retrieval_cache
                ):

                    try:

                        st.session_state[
                            "chat_retrieval_cache"
                        ][
                            cache_key
                        ] = (
                            get_retrieval_signal(
                                question=question,
                                memory_id=(
                                    st.session_state
                                    .chat_memory_id
                                ),
                            )
                        )

                    except Exception:

                        st.session_state[
                            "chat_retrieval_cache"
                        ][
                            cache_key
                        ] = {

                            "grounded": False,

                            "confidence":
                                "Not found",

                            "sources": [],
                        }

                signal = (
                    st.session_state
                    .chat_retrieval_cache[
                        cache_key
                    ]
                )

                # =================================================
                # Grounded answer
                # =================================================

                if signal["grounded"]:

                    st.caption(
                        "Confidence: "
                        f"{signal['confidence']}"
                    )

                    # ---------------------------------------------
                    # Sources
                    # ---------------------------------------------

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

                # =================================================
                # Not found
                # =================================================

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


if question:

    question = question.strip()

    if not question:

        st.toast(
            "Please enter a question.",
            icon="⚠️",
        )

    else:

        try:

            with st.spinner(
                "Memory Vault is thinking..."
            ):

                run_langgraph_agent(
                    question=question,
                    user_id=USER_ID,
                    thread_id=thread_id,
                    memory_id=(
                        st.session_state
                        .chat_memory_id
                    ),
                )

            # =================================================
            # Calculate retrieval confidence for this question.
            # =================================================

            cache_key = get_cache_key(
                thread_id,
                question,
                (
                    st.session_state
                    .chat_memory_id
                ),
            )

            try:

                st.session_state[
                    "chat_retrieval_cache"
                ][
                    cache_key
                ] = get_retrieval_signal(
                    question=question,
                    memory_id=(
                        st.session_state
                        .chat_memory_id
                    ),
                )

            except Exception:

                st.session_state[
                    "chat_retrieval_cache"
                ][
                    cache_key
                ] = {

                    "grounded": False,

                    "confidence":
                        "Not found",

                    "sources": [],
                }

            # =================================================
            # Rerun so the newly persisted messages appear.
            # =================================================

            st.rerun()

        except Exception as error:

            st.toast(
                (
                    "Something went wrong while "
                    f"running the chat: {error}"
                ),
                icon="❌",
            )