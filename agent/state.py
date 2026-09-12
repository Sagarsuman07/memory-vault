from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[
        list,
        add_messages,
    ]

    user_id: str

    question: str

    # Current chat retrieval scope:
    # "all" or "memory".
    scope: str

    # Selected memory when scope == "memory".
    # None means all memories.
    memory_id: str | None

    final_answer: str

    tool_calls: list