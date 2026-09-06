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

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.state import AgentState

from agent.nodes import (
    understand_query,
    agent_node,
    answer_node,
)

from agent.tools import create_tools

from config.settings import settings


# ============================================================
# Persistent LangGraph Checkpointer
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
# Build Graph
# ============================================================

def build_agent_graph(
    user_id: str,
):

    if not user_id or not user_id.strip():

        raise ValueError(
            "User ID cannot be empty."
        )


    # ========================================================
    # Create tools for current user
    # ========================================================

    tools = create_tools(
        user_id=user_id
    )


    # ========================================================
    # Create Tool Node
    # ========================================================

    tool_node = ToolNode(
        tools
    )


    # ========================================================
    # Create State Graph
    # ========================================================

    graph_builder = StateGraph(
        AgentState
    )


    # ========================================================
    # Add Nodes
    # ========================================================

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


    # ========================================================
    # START → Understand Query
    # ========================================================

    graph_builder.add_edge(
        START,
        "understand_query",
    )


    # ========================================================
    # Understand Query → Agent
    # ========================================================

    graph_builder.add_edge(
        "understand_query",
        "agent",
    )


    # ========================================================
    # Agent → Tool or Answer
    # ========================================================

    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            "__end__": "answer",
        },
    )


    # ========================================================
    # Tool → Agent
    # ========================================================

    graph_builder.add_edge(
        "tools",
        "agent",
    )


    # ========================================================
    # Answer → END
    # ========================================================

    graph_builder.add_edge(
        "answer",
        END,
    )


    # ========================================================
    # Compile with Checkpointer
    # ========================================================

    return graph_builder.compile(
        checkpointer=_checkpointer
    )


# ============================================================
# Run LangGraph Agent
# ============================================================

def run_langgraph_agent(
    question: str,
    user_id: str,
    thread_id: str,
):

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )


    if not user_id or not user_id.strip():

        raise ValueError(
            "User ID cannot be empty."
        )


    # ========================================================
    # Validate thread ownership
    # ========================================================

    validate_thread_id(
        user_id=user_id,
        thread_id=thread_id,
    )


    # ========================================================
    # Build graph
    # ========================================================

    graph = build_agent_graph(
        user_id=user_id
    )


    # ========================================================
    # New message
    #
    # The checkpointer restores previous messages
    # belonging to this thread.
    # ========================================================

    input_state = {

        "messages": [
            {
                "role": "user",
                "content": question.strip(),
            }
        ],

        "user_id": user_id,

        "question": question.strip(),

        "final_answer": "",

        "tool_calls": [],
    }


    # ========================================================
    # Thread configuration
    # ========================================================

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }


    # ========================================================
    # Execute graph
    # ========================================================

    result = graph.invoke(
        input_state,
        config,
    )


    # ========================================================
    # Return result
    # ========================================================

    return {

        "answer": result.get(
            "final_answer",
            "",
        ),

        "tool_calls": result.get(
            "tool_calls",
            [],
        ),

        "messages": result.get(
            "messages",
            [],
        ),

        "thread_id": thread_id,
    }


# ============================================================
# Get Conversation State
# ============================================================

def get_thread_state(
    user_id: str,
    thread_id: str,
):

    # ========================================================
    # Validate thread ownership
    # ========================================================

    validate_thread_id(
        user_id=user_id,
        thread_id=thread_id,
    )


    # ========================================================
    # Build graph
    # ========================================================

    graph = build_agent_graph(
        user_id=user_id
    )


    # ========================================================
    # Thread configuration
    # ========================================================

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }


    # ========================================================
    # Retrieve state
    # ========================================================

    return graph.get_state(
        config
    )