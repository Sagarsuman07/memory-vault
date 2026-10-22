"""
Phase 0.3: agent execution limits.

What this module provides
-------------------------
create_limited_tool_node(tools)
    Wraps LangGraph's prebuilt ToolNode (so ToolMessage.artifact,
    parallel execution and error handling keep working) and enforces
    MAX_TOOL_CALLS per request.

    Allowed calls run normally. Calls over the budget are NOT executed:
    they receive a ToolMessage (so the message history stays valid) and
    are recorded in the state fields that answer_node already reads:

        tool_call_count      tool executions actually performed
        tool_limit_reached   True once any call was blocked
        blocked_tool_calls   [{"tool": ..., "arguments": ...}, ...]

route_after_tools(state)
    Once a call was blocked, go straight to answer_node. It already
    explains the limitation and keeps the results that were obtained.
    The agent is not asked again, so a model that insists on more tools
    cannot loop.

get_recursion_limit() / close_dangling_tool_calls()
    MAX_GRAPH_RECURSION is a backstop. If it ever aborts a run, the thread
    is repaired so later questions in the conversation still work.
"""

import logging

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)

from langgraph.prebuilt import ToolNode

from config.settings import settings


logger = logging.getLogger(__name__)


BLOCKED_TOOL_MESSAGE = (
    "This tool was NOT executed because the application tool-call "
    "limit for this request was reached."
)

RECURSION_NOTICE = (
    "I stopped processing this request because the maximum agent "
    "execution limit was reached. Please try asking the question "
    "more specifically."
)


# ============================================================
# Tool node with a per-request budget
# ============================================================

def create_limited_tool_node(tools):

    tool_node = ToolNode(
        tools,
        handle_tool_errors=True,
    )

    valid_tool_names = {
        tool.name
        for tool in tools
    }

    def limited_tool_node(
        state,
        config,
    ):

        messages = state.get("messages", [])

        last_message = messages[-1] if messages else None

        requested = list(
            getattr(last_message, "tool_calls", None) or []
        )

        if not requested:

            return {"messages": []}

        used = int(
            state.get("tool_call_count", 0) or 0
        )

        remaining = max(
            0,
            int(settings.MAX_TOOL_CALLS) - used,
        )

        # ----------------------------------------------------
        # Split the requested calls.
        #
        # - unknown tool name: passed to ToolNode, which answers
        #   with its standard error. Nothing is executed, so it
        #   does not use up the budget.
        # - known tool, budget left: allowed (uses the budget).
        # - known tool, no budget left: blocked.
        # ----------------------------------------------------

        allowed = []

        blocked = []

        budget = remaining

        for call in requested:

            if call["name"] not in valid_tool_names:

                allowed.append(call)

            elif budget > 0:

                allowed.append(call)

                budget -= 1

            else:

                blocked.append(call)

        executed = remaining - budget

        results = {}

        # ----------------------------------------------------
        # Execute only the allowed calls, through the real
        # ToolNode. A copy of the AI message is used so the
        # persisted message is not modified.
        # ----------------------------------------------------

        if allowed:

            limited_message = last_message.model_copy(
                update={"tool_calls": allowed}
            )

            output = tool_node.invoke(
                {
                    **state,
                    "messages": [limited_message],
                },
                config,
            )

            for message in output["messages"]:

                results[message.tool_call_id] = message

        # ----------------------------------------------------
        # Calls over budget are never executed.
        # ----------------------------------------------------

        for call in blocked:

            results[call["id"]] = ToolMessage(
                content=BLOCKED_TOOL_MESSAGE,
                tool_call_id=call["id"],
                name=call["name"],
                status="error",
            )

        # ----------------------------------------------------
        # Every requested call must be answered, otherwise the
        # message history becomes invalid for the chat API.
        # ----------------------------------------------------

        ordered = []

        for call in requested:

            message = results.get(call["id"])

            if message is None:

                message = ToolMessage(
                    content="Tool did not return a result.",
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )

            ordered.append(message)

        update = {
            "messages": ordered,
            "tool_call_count": used + executed,
        }

        if blocked:

            update["tool_limit_reached"] = True

            update["blocked_tool_calls"] = (
                list(state.get("blocked_tool_calls", []) or [])
                + [
                    {
                        "tool": call["name"],
                        "arguments": call.get("args", {}),
                    }
                    for call in blocked
                ]
            )

        return update

    return limited_tool_node


# ============================================================
# Routing after the tools node
# ============================================================

def route_after_tools(
    state,
) -> str:

    if state.get("tool_limit_reached"):

        return "answer"

    return "agent"


# ============================================================
# Recursion limit (backstop)
# ============================================================

_warned = False


def get_recursion_limit() -> int:
    """
    Return MAX_GRAPH_RECURSION and warn once if it is too small for
    the tool budget.

    Worst case path with sequential tool calls:

        understand_query                     1
        (agent + tools) x MAX_TOOL_CALLS     2 x N
        one more agent + tools (blocked)     2
        answer                               1
    """

    global _warned

    limit = int(settings.MAX_GRAPH_RECURSION)

    minimum = 2 * int(settings.MAX_TOOL_CALLS) + 5

    if limit < minimum and not _warned:

        _warned = True

        logger.warning(
            "MAX_GRAPH_RECURSION=%s is lower than %s (2 x "
            "MAX_TOOL_CALLS + 5). The recursion limit can stop a "
            "request before the tool budget is reached.",
            limit,
            minimum,
        )

    return limit


# ============================================================
# Repair after a recursion-limit abort
# ============================================================

def close_dangling_tool_calls(
    graph,
    config,
    notice: str = RECURSION_NOTICE,
):
    """
    Make the thread valid again and store a visible assistant message.

    The last AI message may still contain tool calls that were never
    answered; chat APIs reject such a history, which would make every
    later question in this conversation fail.
    """

    snapshot = graph.get_state(config)

    messages = snapshot.values.get("messages", [])

    additions = []

    last_message = messages[-1] if messages else None

    if (
        last_message is not None
        and getattr(last_message, "type", "") == "ai"
        and getattr(last_message, "tool_calls", None)
    ):

        for call in last_message.tool_calls:

            additions.append(
                ToolMessage(
                    content=BLOCKED_TOOL_MESSAGE,
                    tool_call_id=call["id"],
                    name=call["name"],
                    status="error",
                )
            )

    additions.append(AIMessage(content=notice))

    graph.update_state(
        config,
        {
            "messages": additions,
            "final_answer": notice,
        },
        as_node="answer",
    )