import re
import html
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


from rag.grounding import DEFAULT_DISTANCE_THRESHOLD


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
# Retrieval / Grounding Signal
# ============================================================

def get_empty_retrieval_signal():

    return {
        "grounded": False,
        "confidence": "Not found",
        "sources": [],
        "distance": None,
    }


def parse_grounding_signal_from_artifact(
    artifact,
):
    """
    Parse grounding metadata from ToolMessage.artifact.

    New conversations use artifacts so grounding metadata is never
    placed in the LLM-visible tool content.
    """

    if not isinstance(
        artifact,
        dict,
    ):

        return None

    grounded = artifact.get(
        "grounded"
    )

    if grounded is False:

        return get_empty_retrieval_signal()

    if grounded is not True:

        return None

    sources = []

    for source in artifact.get(
        "sources",
        [],
    ):

        if not isinstance(
            source,
            dict,
        ):

            continue

        source_copy = dict(source)

        try:

            source_copy["chunk_index"] = int(
                source_copy.get(
                    "chunk_index",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            source_copy["chunk_index"] = 0

        try:

            source_copy["distance"] = float(
                source_copy.get(
                    "distance"
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            source_copy["distance"] = None

        sources.append(
            source_copy
        )

    # Resolve titles from user-owned SQLite records only.
    try:

        memories = list_memories(
            USER_ID
        )

        memory_map = {
            memory["id"]: memory
            for memory in memories
        }

        for source in sources:

            source_memory = memory_map.get(
                source["memory_id"]
            )

            source["title"] = (
                source_memory["title"]
                if source_memory is not None
                else source["memory_id"]
            )

    except Exception:

        for source in sources:

            source["title"] = source["memory_id"]

    try:

        distance = float(
            artifact.get(
                "distance"
            )
        )

    except (
        TypeError,
        ValueError,
    ):

        distance = None

    return {
        "grounded": True,
        "confidence": (
            artifact.get(
                "confidence"
            )
            or "Grounded"
        ),
        "sources": sources,
        "distance": distance,
    }


def parse_grounding_signal_from_tool_content(
    tool_content: str,
):
    """
    Backward-compatible parser for conversations created before
    the artifact-based grounding metadata change.
    """

    if not tool_content:

        return None

    content = str(tool_content)

    status_match = re.search(
        r"GROUNDING_STATUS:\s*(GROUNDED|NOT_GROUNDED)",
        content,
        flags=re.IGNORECASE,
    )

    if not status_match:

        return None

    if status_match.group(1).upper() == "NOT_GROUNDED":

        return get_empty_retrieval_signal()

    confidence_match = re.search(
        r"CONFIDENCE:\s*(.+)",
        content,
        flags=re.IGNORECASE,
    )

    distance_match = re.search(
        r"STRONGEST_DISTANCE:\s*([0-9.eE+-]+)",
        content,
        flags=re.IGNORECASE,
    )

    confidence = (
        confidence_match.group(1).strip()
        if confidence_match
        else "Grounded"
    )

    strongest_distance = None

    if distance_match:

        try:

            strongest_distance = float(
                distance_match.group(1)
            )

        except ValueError:

            pass

    sources = []

    # Support both normal newlines and the literal "\n" format
    # produced by the previous version.
    normalized_content = content.replace(
        "\\n",
        "\n",
    )

    result_blocks = re.split(
        r"\n\s*Result\s+\d+\s*\n",
        normalized_content,
    )

    for block in result_blocks[1:]:

        memory_id_match = re.search(
            r"Memory ID:\s*(.+)",
            block,
        )

        memory_type_match = re.search(
            r"Memory Type:\s*(.+)",
            block,
        )

        chunk_match = re.search(
            r"Chunk Index:\s*(.+)",
            block,
        )

        result_distance_match = re.search(
            r"Distance:\s*([0-9.eE+-]+)",
            block,
        )

        if not memory_id_match:

            continue

        source = {
            "memory_id":
                memory_id_match.group(1).strip(),

            "memory_type": (
                memory_type_match.group(1).strip()
                if memory_type_match
                else "unknown"
            ),

            "chunk_index": 0,
            "distance": None,
        }

        if chunk_match:

            try:

                source["chunk_index"] = int(
                    chunk_match.group(1).strip()
                )

            except ValueError:

                pass

        if result_distance_match:

            try:

                source["distance"] = float(
                    result_distance_match.group(1).strip()
                )

            except ValueError:

                pass

        sources.append(source)

    try:

        memories = list_memories(
            USER_ID
        )

        memory_map = {
            memory["id"]: memory
            for memory in memories
        }

        for source in sources:

            source_memory = memory_map.get(
                source["memory_id"]
            )

            source["title"] = (
                source_memory["title"]
                if source_memory is not None
                else source["memory_id"]
            )

    except Exception:

        for source in sources:

            source["title"] = source["memory_id"]

    return {
        "grounded": True,
        "confidence": confidence,
        "sources": sources,
        "distance": strongest_distance,
    }


def _get_used_memory_ids_from_answer(
    message,
):
    """
    Read the memory IDs selected by answer_node.

    answer_node stores them in:
        AIMessage.additional_kwargs["used_memory_ids"]
    """

    additional_kwargs = getattr(
        message,
        "additional_kwargs",
        None,
    )

    if not isinstance(
        additional_kwargs,
        dict,
    ):
        return []

    raw_ids = additional_kwargs.get(
        "used_memory_ids",
        [],
    )

    if not isinstance(
        raw_ids,
        (list, tuple, set),
    ):
        return []

    used_memory_ids = []

    for memory_id in raw_ids:

        if memory_id is None:
            continue

        memory_id = str(
            memory_id
        ).strip()

        if not memory_id:
            continue

        if memory_id not in used_memory_ids:

            used_memory_ids.append(
                memory_id
            )

    return used_memory_ids


def _filter_signal_sources(
    signal,
    used_memory_ids,
):
    """
    Filter the candidate retrieval sources so the UI shows
    only the memories actually used by answer_node.
    """

    if not signal:
        return None

    if not used_memory_ids:
        return {
            **signal,
            "sources": [],
        }

    used_ids = set(
        used_memory_ids
    )

    filtered_sources = []
    seen_memory_ids = set()

    for source in signal.get(
        "sources",
        [],
    ):

        if not isinstance(
            source,
            dict,
        ):
            continue

        memory_id = source.get(
            "memory_id"
        )

        if not memory_id:
            continue

        memory_id = str(
            memory_id
        ).strip()

        if memory_id not in used_ids:
            continue

        # One UI source per memory.
        if memory_id in seen_memory_ids:
            continue

        seen_memory_ids.add(
            memory_id
        )

        filtered_sources.append(
            source
        )

    return {
        **signal,
        "sources": filtered_sources,
    }


def get_tool_grounding_signal_for_answer(
    messages,
    assistant_index: int,
):
    """
    Build the source signal for one final answer.

    Source selection authority:
        answer_node

    The UI never decides which memory is relevant.

    It only resolves the memory IDs selected by answer_node
    into source metadata.

    Sources can come from:

    1. search_memories results in the current turn
    2. historical search_memories results from an earlier turn

    The second case is required when the current answer is
    derived from conversation history instead of performing
    another memory search.
    """

    if (
        messages is None
        or assistant_index is None
        or assistant_index < 0
        or assistant_index >= len(messages)
    ):
        return None

    # ========================================================
    # 1. Read source IDs selected by answer_node.
    # ========================================================

    answer_message = messages[
        assistant_index
    ]

    used_memory_ids = (
        _get_used_memory_ids_from_answer(
            answer_message
        )
    )

    if not used_memory_ids:
        return None

    used_memory_id_set = set(
        used_memory_ids
    )

    # ========================================================
    # 2. Collect source metadata from ALL search_memories
    #    results in the conversation.
    #
    #    This is intentional.
    #
    #    A repeated question may not execute search_memories
    #    again because the answer came from conversation
    #    history.
    # ========================================================

    all_sources = {}

    confidence = "Grounded"
    strongest_distance = None
    found_grounded_result = False

    for message in messages:

        if (
            get_message_type(message)
            != "tool"
        ):
            continue

        tool_name = getattr(
            message,
            "name",
            "",
        )

        if tool_name != "search_memories":
            continue

        # ----------------------------------------------------
        # New artifact-based metadata.
        # ----------------------------------------------------

        artifact_signal = (
            parse_grounding_signal_from_artifact(
                getattr(
                    message,
                    "artifact",
                    None,
                )
            )
        )

        if artifact_signal is not None:

            if not artifact_signal.get(
                "grounded",
                False,
            ):
                continue

            found_grounded_result = True

            artifact_confidence = (
                artifact_signal.get(
                    "confidence"
                )
            )

            if artifact_confidence:
                confidence = (
                    artifact_confidence
                )

            distance = (
                artifact_signal.get(
                    "distance"
                )
            )

            if isinstance(
                distance,
                (int, float),
            ):
                if (
                    strongest_distance is None
                    or distance
                    < strongest_distance
                ):
                    strongest_distance = distance

            for source in artifact_signal.get(
                "sources",
                [],
            ):

                if not isinstance(
                    source,
                    dict,
                ):
                    continue

                memory_id = source.get(
                    "memory_id"
                )

                if not memory_id:
                    continue

                memory_id = str(
                    memory_id
                ).strip()

                # ------------------------------------------------
                # CRITICAL:
                #
                # Only sources explicitly selected by
                # answer_node are allowed into the UI.
                # ------------------------------------------------

                if (
                    memory_id
                    not in used_memory_id_set
                ):
                    continue

                if (
                    memory_id
                    not in all_sources
                ):
                    all_sources[
                        memory_id
                    ] = dict(
                        source
                    )

                else:

                    existing_distance = (
                        all_sources[
                            memory_id
                        ].get(
                            "distance"
                        )
                    )

                    new_distance = (
                        source.get(
                            "distance"
                        )
                    )

                    if (
                        isinstance(
                            new_distance,
                            (int, float),
                        )
                        and (
                            existing_distance
                            is None
                            or new_distance
                            < existing_distance
                        )
                    ):
                        all_sources[
                            memory_id
                        ]["distance"] = (
                            new_distance
                        )

            continue

        # ----------------------------------------------------
        # Backward-compatible old format.
        # ----------------------------------------------------

        content_signal = (
            parse_grounding_signal_from_tool_content(
                get_message_content(
                    message
                )
            )
        )

        if content_signal is None:
            continue

        if not content_signal.get(
            "grounded",
            False,
        ):
            continue

        found_grounded_result = True

        for source in content_signal.get(
            "sources",
            [],
        ):

            if not isinstance(
                source,
                dict,
            ):
                continue

            memory_id = source.get(
                "memory_id"
            )

            if not memory_id:
                continue

            memory_id = str(
                memory_id
            ).strip()

            if (
                memory_id
                not in used_memory_id_set
            ):
                continue

            if (
                memory_id
                not in all_sources
            ):
                all_sources[
                    memory_id
                ] = dict(
                    source
                )

    # ========================================================
    # 3. Resolve selected memory IDs directly from SQLite
    #    if their historical search result is unavailable.
    #
    #    This is a fallback only.
    # ========================================================

    missing_memory_ids = [
        memory_id
        for memory_id in used_memory_ids
        if memory_id
        not in all_sources
    ]

    if missing_memory_ids:

        try:

            memories = list_memories(
                USER_ID
            )

            memory_map = {
                memory["id"]: memory
                for memory in memories
            }

            for memory_id in (
                missing_memory_ids
            ):

                memory = memory_map.get(
                    memory_id
                )

                if memory is None:
                    continue

                all_sources[
                    memory_id
                ] = {
                    "memory_id":
                        memory_id,

                    "memory_type":
                        memory.get(
                            "type"
                        )
                        or memory.get(
                            "memory_type"
                        )
                        or "unknown",

                    "title":
                        memory.get(
                            "title"
                        )
                        or memory_id,

                    "distance":
                        None,
                }

        except Exception:
            pass

    # ========================================================
    # 4. If no selected source can be resolved, don't show
    #    a Sources section.
    # ========================================================

    if not all_sources:
        return None

    # ========================================================
    # 5. Return ONLY answer_node-selected memories.
    # ========================================================

    ordered_sources = []

    for memory_id in used_memory_ids:

        source = all_sources.get(
            memory_id
        )

        if source is None:
            continue

        ordered_sources.append(
            source
        )

    if not ordered_sources:
        return None

    return {
        "grounded": True,
        "confidence": confidence,
        "sources": ordered_sources,
        "distance": strongest_distance,
    }


def get_retrieval_signal(
    question: str,
    memory_id: str | None,
    messages=None,
    assistant_index: int | None = None,
):
    """
    Read grounding/confidence metadata from the actual
    search_memories ToolMessage used by LangGraph.
    """

    if (
        messages is not None
        and assistant_index is not None
    ):

        signal = get_tool_grounding_signal_for_answer(
            messages=messages,
            assistant_index=assistant_index,
        )

        if signal is not None:

            return signal

    return get_empty_retrieval_signal()


def get_cache_key(
    thread_id: str,
    question: str,
    memory_id: str | None,
):

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


def get_or_create_retrieval_signal(
    thread_id: str,
    question: str,
    memory_id: str | None,
    messages=None,
    assistant_index: int | None = None,
):
    """
    Return cached grounding information.

    Grounding is produced by LangGraph's search_memories tool.
    Chroma is not queried again just to display confidence.
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

    signal = get_retrieval_signal(
        question=question,
        memory_id=memory_id,
        messages=messages,
        assistant_index=assistant_index,
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


def open_memory_from_source(memory_id: str):
    """
    Open the Memory Detail page for a source memory.

    The memory ID comes from the user-scoped grounding artifact.
    """

    if not memory_id:
        st.toast(
            "This source memory could not be opened.",
            icon="⚠️",
        )
        return

    # Re-validate ownership before navigation.
    memory = get_existing_memory(memory_id)

    if memory is None:
        st.toast(
            "This memory no longer exists.",
            icon="⚠️",
        )
        return

    # Store the selected memory in session state.
    # Memory Detail uses this value when opening the page.
    st.session_state["selected_memory_id"] = memory_id

    # Also keep it in the URL/query parameters.
    st.query_params["memory_id"] = memory_id

    st.switch_page(
        "pages/memory_detail.py"
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

def get_final_ai_message_for_turn(
    messages,
    human_index,
):
    """
    Return the last non-empty AI message belonging to the
    current user turn.

    A user turn ends when the next human message starts.

    This is important because LangGraph may persist:
        - tool-calling AI messages
        - tool messages
        - final AI answer message

    Only the final non-empty AI message should be displayed
    to the user.
    """

    final_ai_index = None

    for index in range(
        human_index + 1,
        len(messages),
    ):
        message_type = get_message_type(
            messages[index]
        )

        # Stop when the next user message begins.
        if message_type == "human":
            break

        if message_type != "ai":
            continue

        content = get_message_content(
            messages[index]
        ).strip()

        if not content:
            continue

        # Keep updating this so the LAST non-empty AI
        # message becomes the final answer.
        final_ai_index = index

    if final_ai_index is None:
        return None

    return (
        final_ai_index,
        messages[final_ai_index],
    )


def render_sources(
    signal,
    assistant_message=None,
    assistant_index=None,
):
    """
    Render ONLY the memory sources actually used by
    answer_node.

    Sources are displayed at memory level.
    Duplicate memory IDs are removed as a final UI safeguard.
    """

    sources = []

    # --------------------------------------------------------
    # Preferred source:
    # answer_node's retrieval_sources
    # --------------------------------------------------------

    if signal:

        sources = list(
            signal.get(
                "sources",
                [],
            )
        )

    # --------------------------------------------------------
    # Nothing to display.
    # --------------------------------------------------------

    if not sources:
        return

    # --------------------------------------------------------
    # Final UI-level deduplication.
    #
    # This is only a safety net. The actual source selection
    # has already happened inside answer_node.
    # --------------------------------------------------------

    unique_sources = []

    seen_memory_ids = set()

    for source in sources:

        if not isinstance(
            source,
            dict,
        ):
            continue

        memory_id = source.get(
            "memory_id"
        )

        if not memory_id:
            continue

        if memory_id in seen_memory_ids:
            continue

        seen_memory_ids.add(
            memory_id
        )

        unique_sources.append(
            source
        )

    if not unique_sources:
        return

    # --------------------------------------------------------
    # Sources UI
    # --------------------------------------------------------

    with st.expander(
        "Sources"
    ):

        for source_index, source in enumerate(
            unique_sources
        ):

            source_memory_id = source.get(
                "memory_id"
            )

            source_title = (
                source.get(
                    "title"
                )
                or source_memory_id
                or "Untitled Memory"
            )

            source_type = (
                source.get(
                    "memory_type"
                )
                or "unknown"
            )

            source_type = str(
                source_type
            ).capitalize()

            source_col1, source_col2, source_col3 = (
                st.columns(
                    [0.70, 0.18, 0.12],
                    vertical_alignment="center",
                )
            )

            # ------------------------------------------------
            # Source title
            # ------------------------------------------------

            with source_col1:

                safe_title = html.escape(
                    str(source_title),
                    quote=True,
                )

                st.markdown(
                    f'<div title="{safe_title}" '
                    'style="'
                    'white-space: nowrap; '
                    'overflow: hidden; '
                    'text-overflow: ellipsis; '
                    'width: 100%; '
                    'line-height: 2rem;'
                    '">'
                    f"{safe_title}"
                    "</div>",
                    unsafe_allow_html=True,
                )

            # ------------------------------------------------
            # Source type
            # ------------------------------------------------

            with source_col2:

                st.caption(
                    source_type
                )

            # ------------------------------------------------
            # Open memory
            # ------------------------------------------------

            with source_col3:

                if st.button(
                    "Open",
                    key=(
                        f"source_open_"
                        f"{assistant_index}_"
                        f"{source_index}_"
                        f"{source_memory_id}"
                    ),
                    use_container_width=True,
                ):

                    open_memory_from_source(
                        source_memory_id
                    )

# ============================================================
# Render Conversation
# ============================================================

index = 0

while index < len(
    conversation_messages
):

    message = conversation_messages[
        index
    ]

    message_type = get_message_type(
        message
    )

    # ========================================================
    # User Message
    # ========================================================

    if message_type != "human":

        index += 1
        continue

    question = get_message_content(
        message
    ).strip()

    if question:

        with st.chat_message(
            "user"
        ):

            st.write(
                question
            )

    # ========================================================
    # Find ONLY the final AI response
    # for this user turn.
    # ========================================================

    final_ai = (
        get_final_ai_message_for_turn(
            conversation_messages,
            index,
        )
    )

    if final_ai is None:

        index += 1
        continue

    assistant_index, assistant_message = (
        final_ai
    )

    content = get_message_content(
        assistant_message
    ).strip()

    if not content:

        index += 1
        continue

    # ========================================================
    # Final Assistant Response
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            content
        )

        # ----------------------------------------------------
        # Retrieval / grounding information
        # ----------------------------------------------------

        signal = (
            get_or_create_retrieval_signal(
                thread_id=thread_id,
                question=question,
                memory_id=(
                    st.session_state.chat_memory_id
                    if (
                        st.session_state.chat_scope
                        == "memory"
                    )
                    else None
                ),
                messages=conversation_messages,
                assistant_index=assistant_index,
            )
        )

        # ----------------------------------------------------
        # Grounded response
        # ----------------------------------------------------

        if signal.get(
            "grounded",
            False,
        ):

            st.caption(
                "Confidence: "
                f"{signal.get('confidence', 'Grounded')}"
            )

            render_sources(
                signal=signal,
                assistant_index=assistant_index,
            )

        # ----------------------------------------------------
        # Not found response
        # ----------------------------------------------------

        elif (
            NOT_FOUND_MESSAGE
            in content
        ):

            st.caption(
                "Not found"
            )

    # ========================================================
    # Move to the next user turn.
    #
    # We deliberately jump to the final AI message instead
    # of processing intermediate tool/AI messages individually.
    # ========================================================

    index = assistant_index + 1

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
            # Read the persisted LangGraph state and extract
            # grounding metadata from the actual search_memories
            # ToolMessage.
            #
            # No second semantic retrieval is performed.
            # ------------------------------------------------

            updated_thread_state = get_thread_state(
                user_id=USER_ID,
                thread_id=thread_id,
            )

            updated_messages = (
                updated_thread_state.values.get(
                    "messages",
                    [],
                )
            )

            assistant_index = None

            for message_index in range(
                len(updated_messages) - 1,
                -1,
                -1,
            ):

                if (
                    get_message_type(
                        updated_messages[message_index]
                    )
                    == "ai"
                ):

                    content = get_message_content(
                        updated_messages[message_index]
                    )

                    if content.strip():

                        assistant_index = message_index

                        break

            get_or_create_retrieval_signal(
                thread_id=thread_id,
                question=question,
                memory_id=current_memory_id,
                messages=updated_messages,
                assistant_index=assistant_index,
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