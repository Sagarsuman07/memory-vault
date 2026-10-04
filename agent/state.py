from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ============================================================
# Backend-Controlled Memory Scope
# ============================================================

class MemoryScope(TypedDict):
    """
    Backend-controlled authorization scope for memory access.

    IMPORTANT:
    - Created by the application.
    - Never created or modified by the LLM.
    - Used by tools to enforce memory isolation.
    """

    user_id: str

    # "all"    -> all memories belonging to user_id
    # "memory" -> only memory_ids belonging to user_id
    scope_type: str

    # None for global scope.
    # List of authorized memory IDs for memory scope.
    memory_ids: list[str] | None


# ============================================================
# Agent State
# ============================================================

class AgentState(TypedDict):

    # ========================================================
    # Conversation
    # ========================================================

    messages: Annotated[
        list,
        add_messages,
    ]

    # ========================================================
    # Request Information
    # ========================================================

    user_id: str

    question: str

    # "all" or "memory"
    scope: str

    # Selected memory when scope == "memory"
    memory_id: str | None

    # Backend-controlled authorization scope
    memory_scope: MemoryScope

    # ========================================================
    # Final Answer
    # ========================================================

    final_answer: str

    # ========================================================
    # Retrieval Sources
    # ========================================================

    # Memory-level sources actually returned by
    # search_memories.
    #
    # One entry per memory.
    #
    # Example:
    #
    # [
    #     {
    #         "memory_id": "...",
    #         "memory_type": "document",
    #         "title": "Hotel Booking",
    #         "distance": 0.21
    #     }
    # ]
    retrieval_sources: list

    # ========================================================
    # Agent Routing
    # ========================================================

    # True when the latest agent response requested tools.
    agent_has_tool_calls: bool

    # ========================================================
    # Tool Tracking
    # ========================================================

    # All requested tool calls.
    tool_calls: list

    # Number of tools actually executed.
    tool_call_count: int

    # True if a tool call was blocked.
    tool_limit_reached: bool

    # Tool calls requested but not executed.
    blocked_tool_calls: list