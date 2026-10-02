import ast
import operator as op

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from config.settings import settings

from database.repositories import (
    get_memory as get_memory_record,
)

from rag.retriever import retrieve
from rag.grounding import (
    DEFAULT_DISTANCE_THRESHOLD,
    get_grounded_results,
)

from agent.state import MemoryScope


# ============================================================
# Safe Calculator
# ============================================================

_ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _safe_calculate(node):
    """
    Recursively evaluate a restricted arithmetic AST.
    """

    if isinstance(node, ast.Constant):

        if isinstance(
            node.value,
            (int, float),
        ):
            return node.value

        raise ValueError(
            "Only numbers are allowed."
        )

    if isinstance(node, ast.UnaryOp):

        operator = _ALLOWED_OPERATORS.get(
            type(node.op)
        )

        if operator is None:

            raise ValueError(
                "Unsupported unary operator."
            )

        return operator(
            _safe_calculate(
                node.operand
            )
        )

    if isinstance(node, ast.BinOp):

        operator = _ALLOWED_OPERATORS.get(
            type(node.op)
        )

        if operator is None:

            raise ValueError(
                "Unsupported operator."
            )

        left = _safe_calculate(
            node.left
        )

        right = _safe_calculate(
            node.right
        )

        # ----------------------------------------
        # Prevent dangerous exponentiation
        # ----------------------------------------

        if (
            isinstance(node.op, ast.Pow)
            and abs(right) > 10
        ):

            raise ValueError(
                "Exponent is too large."
            )

        return operator(
            left,
            right,
        )

    raise ValueError(
        "Only arithmetic expressions are supported."
    )


def calculate_expression(
    expression: str
):

    if not expression or not expression.strip():

        raise ValueError(
            "Expression cannot be empty."
        )

    try:

        tree = ast.parse(
            expression,
            mode="eval",
        )

        result = _safe_calculate(
            tree.body
        )

        return result

    except ZeroDivisionError:

        raise ValueError(
            "Cannot divide by zero."
        )

    except (
        SyntaxError,
        ValueError,
        TypeError,
    ) as error:

        raise ValueError(
            f"Invalid arithmetic expression: {error}"
        )


# ============================================================
# Scope Validation
# ============================================================

def _validate_scope(
    scope: MemoryScope,
):
    """
    Validate the backend-created memory scope.

    This function is intentionally independent of the LLM.

    The LLM cannot create or modify this scope because the scope
    is supplied by the application when the tools are created.
    """

    if not isinstance(scope, dict):

        raise ValueError(
            "Invalid memory scope."
        )

    user_id = scope.get("user_id")

    if not isinstance(user_id, str) or not user_id.strip():

        raise ValueError(
            "Memory scope user ID cannot be empty."
        )

    scope_type = scope.get(
        "scope_type"
    )

    if scope_type not in {
        "all",
        "memory",
    }:

        raise ValueError(
            "Invalid memory scope type."
        )

    memory_ids = scope.get(
        "memory_ids"
    )

    if scope_type == "all":

        if memory_ids is not None:

            raise ValueError(
                "Global scope cannot contain memory IDs."
            )

    else:

        if not isinstance(
            memory_ids,
            list,
        ) or not memory_ids:

            raise ValueError(
                "Memory scope must contain at least one memory ID."
            )

        for memory_id in memory_ids:

            if (
                not isinstance(
                    memory_id,
                    str,
                )
                or not memory_id.strip()
            ):

                raise ValueError(
                    "Memory scope contains an invalid memory ID."
                )


def _is_memory_allowed(
    scope: MemoryScope,
    memory_id: str,
) -> bool:
    """
    Determine whether a memory ID is authorized by the
    backend-controlled scope.
    """

    if not memory_id:
        return False

    # Global scope:
    # authorization is handled by user_id filtering.
    if scope["scope_type"] == "all":
        return True

    # Memory-scoped chat:
    # only explicitly selected memories are allowed.
    return memory_id in (
        scope["memory_ids"] or []
    )


# ============================================================
# Create Tools
# ============================================================

def create_tools(
    scope: MemoryScope,
):

    # --------------------------------------------------------
    # Validate backend-controlled scope once when tools are
    # created.
    # --------------------------------------------------------

    _validate_scope(
        scope
    )

    user_id = scope["user_id"]

    # ========================================================
    # Tool 1 — Search Memories
    # ========================================================

    @tool(response_format="content_and_artifact")
    def search_memories(
        query: str,
    ):
        """
        Search the user's saved memories using semantic search.

        IMPORTANT:
        The search scope is controlled entirely by the backend.

        The LLM can provide only the search query.
        It cannot provide user_id or memory_ids.
        """

        if not query or not query.strip():

            return (
                "No search query was provided.",
                {
                    "grounded": False,
                    "confidence": "Not found",
                    "distance": None,
                    "sources": [],
                },
            )

        # ----------------------------------------------------
        # Convert backend scope into retrieval parameters.
        # ----------------------------------------------------

        selected_memory_id = None

        if scope["scope_type"] == "memory":

            memory_ids = (
                scope["memory_ids"] or []
            )

            # V1 currently supports one selected memory.
            # We deliberately keep the retrieval API compatible.
            selected_memory_id = memory_ids[0]

        results = retrieve(
            question=query,
            user_id=user_id,
            top_k=5,
            memory_id=selected_memory_id,
        )

        if not results:

            return (
                "No grounded memory evidence was found for this query. "
                "Answer exactly: This information wasn't found in your memory.",
                {
                    "grounded": False,
                    "confidence": "Not found",
                    "distance": None,
                    "sources": [],
                },
            )

        # --------------------------------------------------------
        # Grounding gate
        # --------------------------------------------------------

        grounded_results = get_grounded_results(
            results
        )

        if not grounded_results:

            return (
                "No grounded memory evidence was found for this query. "
                "Answer exactly: This information wasn't found in your memory.",
                {
                    "grounded": False,
                    "confidence": "Not found",
                    "distance": None,
                    "sources": [],
                    "threshold": DEFAULT_DISTANCE_THRESHOLD,
                },
            )

        strongest_distance = min(
            distance
            for _, distance in grounded_results
        )

        if strongest_distance <= 0.4:
            confidence = "High retrieval confidence"
        else:
            confidence = "Medium retrieval confidence"

        # --------------------------------------------------------
        # Only evidence is exposed to the LLM.
        # --------------------------------------------------------

        formatted_results = []
        sources = []

        for rank, (
            document,
            distance,
        ) in enumerate(
            grounded_results,
            start=1,
        ):

            memory_id_value = document.metadata[
                "memory_id"
            ]

            # ------------------------------------------------
            # Defense-in-depth scope check.
            #
            # Even though retrieve() already filters by
            # user_id/memory_id, verify the returned memory
            # against the authorization scope before exposing
            # it to the model.
            # ------------------------------------------------

            if not _is_memory_allowed(
                scope,
                memory_id_value,
            ):
                continue

            memory_type = document.metadata[
                "memory_type"
            ]

            chunk_index = document.metadata[
                "chunk_index"
            ]

            formatted_results.append(
                f"""
Evidence {rank}

Content:
{document.page_content}
""".strip()
            )

            sources.append(
                {
                    "memory_id": memory_id_value,
                    "memory_type": memory_type,
                    "chunk_index": int(chunk_index),
                    "distance": float(distance),
                }
            )

        # --------------------------------------------------------
        # Defense-in-depth:
        # if retrieval returned results but none survived the
        # authorization check, do not expose them.
        # --------------------------------------------------------

        if not formatted_results:

            return (
                "No authorized memory evidence was found for this query. "
                "Answer exactly: This information wasn't found in your memory.",
                {
                    "grounded": False,
                    "confidence": "Not found",
                    "distance": None,
                    "sources": [],
                },
            )

        artifact = {
            "grounded": True,
            "confidence": confidence,
            "distance": float(strongest_distance),
            "threshold": DEFAULT_DISTANCE_THRESHOLD,
            "sources": sources,
        }

        return (
            "\n\n".join(formatted_results),
            artifact,
        )

    # ========================================================
    # Tool 2 — Get Memory
    # ========================================================

    @tool
    def get_memory(
        memory_id_requested: str,
    ) -> str:
        """
        Get detailed information about one saved memory.

        The requested memory ID comes from the LLM, but the
        backend-controlled scope decides whether access is allowed.
        """

        if (
            not memory_id_requested
            or not memory_id_requested.strip()
        ):

            return (
                "Memory ID cannot be empty."
            )

        memory_id_requested = (
            memory_id_requested.strip()
        )

        # ----------------------------------------------------
        # Scope security
        # ----------------------------------------------------

        if not _is_memory_allowed(
            scope,
            memory_id_requested,
        ):

            return (
                "Memory not found."
            )

        # ----------------------------------------------------
        # Ownership security
        #
        # Even in global scope, the database lookup is always
        # performed with the backend-controlled user_id.
        # ----------------------------------------------------

        memory = get_memory_record(
            memory_id=memory_id_requested,
            user_id=user_id,
        )

        if memory is None:

            return (
                "Memory not found."
            )

        return f"""
Memory ID:
{memory["id"]}

Title:
{memory["title"]}

Memory Type:
{memory["memory_type"]}

File Name:
{memory["file_name"]}

Summary:
{memory["summary"] or "No summary available."}

Extracted Content:
{memory["extracted_text"]}
""".strip()

    # ========================================================
    # Tool 3 — Calculator
    # ========================================================

    @tool
    def calculate(
        expression: str,
    ) -> str:
        """
        Calculate a mathematical expression.

        Use this tool when arithmetic is required.
        Do not perform complex arithmetic manually.
        """

        try:

            result = calculate_expression(
                expression
            )

            return (
                f"Result: {result}"
            )

        except ValueError as error:

            return (
                f"Calculation error: {error}"
            )

    # ========================================================
    # Tool 4 — Web Search
    # ========================================================

    tavily_tool = None

    if settings.TAVILY_API_KEY:

        tavily_tool = TavilySearch(
            max_results=5,
            topic="general",
        )

    @tool
    def search_web(
        query: str,
    ) -> str:
        """
        Search the internet for current or external information.

        Web search does not receive or modify memory scope.
        """

        if not settings.TAVILY_API_KEY:

            return (
                "Web search is not configured. "
                "TAVILY_API_KEY is missing."
            )

        if not query or not query.strip():

            return (
                "Web search query cannot be empty."
            )

        try:

            results = tavily_tool.invoke(
                {
                    "query": query
                }
            )

            return str(
                results
            )

        except Exception as error:

            return (
                f"Web search failed: {error}"
            )

    return [
        search_memories,
        get_memory,
        calculate,
        search_web,
    ]