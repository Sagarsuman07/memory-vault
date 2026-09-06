from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.prebuilt import (
    ToolNode,
    tools_condition,
)

from agent.state import AgentState

from agent.nodes import (
    understand_query,
    agent_node,
    answer_node,
)

from agent.tools import create_tools


# ============================================================
# Build Graph
# ============================================================

def build_agent_graph(
    user_id: str,
):

    # ========================================================
    # Create tools for the current user
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
    # Compile
    # ========================================================

    return graph_builder.compile()


# ============================================================
# Run Agent
# ============================================================

def run_langgraph_agent(
    question: str,
    user_id: str,
):

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )


    graph = build_agent_graph(
        user_id=user_id
    )


    initial_state = {
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


    result = graph.invoke(
        initial_state
    )


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
    }