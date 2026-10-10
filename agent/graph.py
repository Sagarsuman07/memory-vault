import sqlite3
import uuid

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from langgraph.checkpoint.sqlite import (
    SqliteSaver,
)

from agent.state import (
    AgentState,
    MemoryScope,
)

from agent.nodes import (
    understand_query,
    agent_node,
    answer_node,
)

from agent.tools import create_tools

from agent.limits import (
    close_dangling_tool_calls,
    create_limited_tool_node,
    get_recursion_limit,
    route_after_tools,
)

from langgraph.errors import GraphRecursionError

from config.settings import settings

from database.repositories import (
    get_memory as get_memory_record,
)


# ============================================================
# Persistent Checkpointer
# ============================================================

_checkpoint_connection = sqlite3.connect(
    settings.CHECKPOINT_DATABASE_PATH,
    check_same_thread=False,
)

_checkpointer = SqliteSaver(
    _checkpoint_connection
)

_checkpointer.setup()


# ============================================================
# Thread ID
# ============================================================

def create_thread_id(
    user_id: str,
) -> str:

    if not user_id or not user_id.strip():

        raise ValueError(
            "User ID cannot be empty."
        )

    return (
        f"{user_id}:"
        f"{uuid.uuid4()}"
    )


def get_default_thread_id(
    user_id: str,
) -> str:

    if not user_id or not user_id.strip():

        raise ValueError(
            "User ID cannot be empty."
        )

    return (
        f"{user_id}:default"
    )


# ============================================================
# Thread Security
# ============================================================

def validate_thread_id(
    user_id: str,
    thread_id: str,
):

    if not user_id or not user_id.strip():

        raise ValueError(
            "User ID cannot be empty."
        )

    if not thread_id or not thread_id.strip():

        raise ValueError(
            "Thread ID cannot be empty."
        )

    expected_prefix = (
        f"{user_id}:"
    )

    if not thread_id.startswith(
        expected_prefix
    ):

        raise ValueError(
            "Thread does not belong "
            "to the current user."
        )


# ============================================================
# Memory Scope
# ============================================================

def build_memory_scope(
    user_id: str,
    memory_id: str | None = None,
) -> MemoryScope:

    if not user_id or not user_id.strip():

        raise ValueError(
            "User ID cannot be empty."
        )

    user_id = user_id.strip()

    # --------------------------------------------------------
    # Global scope
    # --------------------------------------------------------

    if memory_id is None:

        return {
            "user_id": user_id,
            "scope_type": "all",
            "memory_ids": None,
        }

    memory_id = memory_id.strip()

    if not memory_id:

        raise ValueError(
            "Memory ID cannot be empty."
        )

    # --------------------------------------------------------
    # Validate ownership
    # --------------------------------------------------------

    memory = get_memory_record(
        memory_id=memory_id,
        user_id=user_id,
    )

    if memory is None:

        raise ValueError(
            "Selected memory does not belong "
            "to the current user."
        )

    return {
        "user_id": user_id,
        "scope_type": "memory",
        "memory_ids": [
            memory_id
        ],
    }


# ============================================================
# Build Graph
# ============================================================

def build_agent_graph(
    user_id: str,
    memory_id: str | None = None,
):

    memory_scope = build_memory_scope(
        user_id=user_id,
        memory_id=memory_id,
    )

    tools = create_tools(
        scope=memory_scope
    )

    tool_node = create_limited_tool_node(
        tools
    )

    graph_builder = StateGraph(
        AgentState
    )

    # --------------------------------------------------------
    # Nodes
    # --------------------------------------------------------

    graph_builder.add_node(
        "understand_query",
        understand_query,
    )

    graph_builder.add_node(
        "agent",
        agent_node,
    )

    graph_builder.add_node(
        "tools",
        tool_node,
    )

    graph_builder.add_node(
        "answer",
        answer_node,
    )

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    graph_builder.add_edge(
        START,
        "understand_query",
    )

    graph_builder.add_edge(
        "understand_query",
        "agent",
    )

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "answer",
        },
    )

    # After the tools run: if a call was blocked by MAX_TOOL_CALLS,
    # go straight to answer_node (it explains the limitation and keeps
    # the results already obtained). Otherwise ask the agent again.
    graph_builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "agent": "agent",
            "answer": "answer",
        },
    )

    graph_builder.add_edge(
        "answer",
        END,
    )

    return graph_builder.compile(
        checkpointer=_checkpointer
    )


# ============================================================
# Run Agent
# ============================================================

def run_langgraph_agent(
    question: str,
    user_id: str,
    thread_id: str,
    memory_id: str | None = None,
):

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )

    if not user_id or not user_id.strip():

        raise ValueError(
            "User ID cannot be empty."
        )

    if (
        memory_id is not None
        and not memory_id.strip()
    ):

        raise ValueError(
            "Memory ID cannot be empty."
        )

    validate_thread_id(
        user_id=user_id,
        thread_id=thread_id,
    )

    memory_scope = build_memory_scope(
        user_id=user_id,
        memory_id=memory_id,
    )

    graph = build_agent_graph(
        user_id=user_id,
        memory_id=memory_id,
    )

    input_state = {

        "messages": [
            {
                "role": "user",
                "content":
                    question.strip(),
            }
        ],

        "user_id":
            user_id,

        "question":
            question.strip(),

        "scope":
            (
                "memory"
                if memory_id is not None
                else "all"
            ),

        "memory_id":
            memory_id,

        "memory_scope":
            memory_scope,

        "final_answer":
            "",

        "retrieval_sources":
            [],

        "tool_calls":
            [],

        "tool_call_count":
            0,

        "tool_limit_reached":
            False,

        "blocked_tool_calls":
            [],

        "agent_has_tool_calls":
            False,
    }

    config = {
        "configurable": {
            "thread_id":
                thread_id,
        },
        "metadata": {
            "user_id": user_id,
            "scope": (
                "memory"
                if memory_id is not None
                else "all"
            ),
            "memory_id": memory_id,
            "max_tool_calls": settings.MAX_TOOL_CALLS,
        },
        "tags": [
            "memory-vault",
            "phase-0.4",
        ],

        # Backstop only. MAX_TOOL_CALLS normally stops a request first.
        "recursion_limit":
            get_recursion_limit(),
    }

    try:

        result = graph.invoke(
            input_state,
            config,
        )

    except GraphRecursionError:

        # Repair the thread (answer dangling tool calls, store a
        # visible assistant message) so later questions still work.
        close_dangling_tool_calls(
            graph,
            config,
        )

        result = dict(
            graph.get_state(config).values
        )

    return {
        "answer":
            result.get(
                "final_answer",
                "",
            ),

        "tool_calls":
            result.get(
                "tool_calls",
                [],
            ),

        "messages":
            result.get(
                "messages",
                [],
            ),

        "thread_id":
            thread_id,

        "scope":
            (
                "memory"
                if memory_id is not None
                else "all"
            ),

        "memory_id":
            memory_id,

        "memory_scope":
            memory_scope,

        # IMPORTANT:
        # Chat UI should use this directly.
        "sources":
            result.get(
                "retrieval_sources",
                [],
            ),

        "retrieval_sources":
            result.get(
                "retrieval_sources",
                [],
            ),

        "tool_call_count":
            result.get(
                "tool_call_count",
                0,
            ),

        "tool_limit_reached":
            result.get(
                "tool_limit_reached",
                False,
            ),

        "blocked_tool_calls":
            result.get(
                "blocked_tool_calls",
                [],
            ),
    }


# ============================================================
# Get Conversation State
# ============================================================

def get_thread_state(
    user_id: str,
    thread_id: str,
):

    validate_thread_id(
        user_id=user_id,
        thread_id=thread_id,
    )

    graph = build_agent_graph(
        user_id=user_id
    )

    config = {
        "configurable": {
            "thread_id":
                thread_id,
        }
    }

    return graph.get_state(
        config
    )


# ============================================================
# Conversation History
# ============================================================

def get_conversation_history(
    user_id: str,
):

    if not user_id or not user_id.strip():

        raise ValueError(
            "User ID cannot be empty."
        )

    user_prefix = (
        f"{user_id}:"
    )

    conversations = []

    try:

        checkpoint_items = list(
            _checkpointer.list(
                None,
                limit=1000,
            )
        )

    except Exception as error:

        raise RuntimeError(
            "Unable to load conversation history."
        ) from error

    thread_ids = []

    for item in checkpoint_items:

        try:

            config = item.config

            configurable = (
                config.get(
                    "configurable",
                    {},
                )
            )

            thread_id = (
                configurable.get(
                    "thread_id"
                )
            )

        except Exception:

            continue

        if not thread_id:
            continue

        if not thread_id.startswith(
            user_prefix
        ):
            continue

        if thread_id not in thread_ids:

            thread_ids.append(
                thread_id
            )

    for thread_id in thread_ids:

        try:

            thread_state = get_thread_state(
                user_id=user_id,
                thread_id=thread_id,
            )

            values = (
                thread_state.values
                if thread_state is not None
                else {}
            )

            messages = values.get(
                "messages",
                [],
            )

            if not messages:
                continue

            first_question = None

            for message in messages:

                if getattr(
                    message,
                    "type",
                    "",
                ) != "human":

                    continue

                content = getattr(
                    message,
                    "content",
                    "",
                )

                if isinstance(
                    content,
                    str,
                ):

                    content = content.strip()

                else:

                    content = str(
                        content
                    ).strip()

                if content:

                    first_question = (
                        content
                    )

                    break

            if not first_question:
                continue

            conversations.append(
                {
                    "thread_id":
                        thread_id,

                    "title":
                        first_question,

                    "scope":
                        values.get(
                            "scope",
                            "all",
                        ),

                    "memory_id":
                        values.get(
                            "memory_id"
                        ),

                    "message_count":
                        len(messages),
                }
            )

        except Exception:

            continue

    conversations.reverse()

    return conversations