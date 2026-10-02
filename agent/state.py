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
    - This object is created by the application.
    - The LLM never supplies or modifies these values.
    - Tools use this scope to enforce user/memory isolation.
    """

    user_id: str

    # "all"    -> all memories belonging to user_id
    # "memory" -> only memory_ids belonging to user_id
    scope_type: str

    # None means all memories for the user.
    # Otherwise only these memory IDs are authorized.
    memory_ids: list[str] | None


# ============================================================
# Agent State
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[
        list,
        add_messages,
    ]

    # --------------------------------------------------------
    # Current authenticated/application user.
    # --------------------------------------------------------

    user_id: str

    # --------------------------------------------------------
    # Current question.
    # --------------------------------------------------------

    question: str

    # --------------------------------------------------------
    # Backward-compatible scope representation.
    #
    # Existing UI/history code uses these fields, so we keep
    # them during Phase 0.1 instead of unnecessarily changing
    # the rest of the application.
    # --------------------------------------------------------

    # "all" or "memory"
    scope: str

    # Selected memory when scope == "memory".
    memory_id: str | None

    # --------------------------------------------------------
    # Formal backend-controlled authorization scope.
    # --------------------------------------------------------

    memory_scope: MemoryScope

    # --------------------------------------------------------
    # Final response.
    # --------------------------------------------------------

    final_answer: str

    # --------------------------------------------------------
    # Tool execution trace.
    # --------------------------------------------------------

    tool_calls: list