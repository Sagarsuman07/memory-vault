import ast
import operator as op

from langchain_core.tools import tool
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
# Phase 0.2 — Tool Output Limits
# ============================================================

# Number of retrieved evidence chunks returned by search_memories.
MAX_MEMORY_SEARCH_CHUNKS = 5

# Maximum characters from one retrieved chunk.
MAX_CHARS_PER_MEMORY_CHUNK = 1500

# Maximum characters across all retrieved memory evidence.
MAX_MEMORY_EVIDENCE_CHARS = 7500

# Maximum extracted-content characters returned by get_memory.
MAX_MEMORY_DETAIL_CHARS = 4000

# Maximum characters returned by search_web.
MAX_WEB_RESULT_CHARS = 6000

# Marker used when tool output is truncated.
TRUNCATION_MARKER = "\n...[truncated]"


# ============================================================
# Generic Text Limiter
# ============================================================

def _truncate_text(
    text: str,
    max_chars: int,
) -> str:
    """
    Limit tool output to a fixed number of characters.

    The limit is intentionally character-based in Phase 0.2.
    Phase 2 will introduce token-aware budgets.

    Args:
        text: Text to limit.
        max_chars: Maximum allowed characters.

    Returns:
        Original text if within the limit, otherwise a
        truncated version with an explicit marker.
    """

    if not text:
        return ""

    if max_chars <= 0:
        return ""

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

    # Keep the marker inside the requested limit.
    content_limit = max_chars - len(
        TRUNCATION_MARKER
    )

    if content_limit <= 0:
        return TRUNCATION_MARKER[:max_chars]

    return (
        text[:content_limit].rstrip()
        + TRUNCATION_MARKER
    )


# ============================================================
# Memory Evidence Limiter
# ============================================================

def _build_memory_evidence(
    grounded_results,
    scope: MemoryScope,
):
    """
    Convert grounded retrieval results into bounded tool output.

    Limits:
        - Maximum number of chunks.
        - Maximum characters per chunk.
        - Maximum total evidence characters.

    Returns:
        formatted evidence,
        source metadata,
        strongest distance
    """

    formatted_results = []
    sources = []

    total_chars = 0

    strongest_distance = None

    evidence_rank = 0

    for document, distance in grounded_results:

        # ----------------------------------------------------
        # Enforce chunk count.
        # ----------------------------------------------------

        if evidence_rank >= MAX_MEMORY_SEARCH_CHUNKS:
            break

        memory_id_value = document.metadata.get(
            "memory_id"
        )

        # ----------------------------------------------------
        # Defense-in-depth scope check.
        # ----------------------------------------------------

        if not memory_id_value:
            continue

        if not _is_memory_allowed(
            scope,
            memory_id_value,
        ):
            continue

        memory_type = document.metadata.get(
            "memory_type",
            "unknown",
        )

        chunk_index = document.metadata.get(
            "chunk_index",
            0,
        )

        # ----------------------------------------------------
        # Calculate remaining total evidence budget.
        # ----------------------------------------------------

        remaining_chars = (
            MAX_MEMORY_EVIDENCE_CHARS
            - total_chars
        )

        if remaining_chars <= 0:
            break

        # ----------------------------------------------------
        # Apply per-chunk limit AND total limit.
        # ----------------------------------------------------

        chunk_limit = min(
            MAX_CHARS_PER_MEMORY_CHUNK,
            remaining_chars,
        )

        content = _truncate_text(
            document.page_content,
            chunk_limit,
        )

        if not content:
            continue

        evidence_rank += 1

        evidence = (
            f"Evidence {evidence_rank}\n\n"
            f"Content:\n"
            f"{content}"
        )

        # ----------------------------------------------------
        # Protect total output budget.
        #
        # The evidence wrapper itself also consumes chars,
        # so check the complete formatted evidence.
        # ----------------------------------------------------

        evidence_remaining = (
            MAX_MEMORY_EVIDENCE_CHARS
            - total_chars
        )

        if len(evidence) > evidence_remaining:

            evidence = _truncate_text(
                evidence,
                evidence_remaining,
            )

        if not evidence:
            break

        formatted_results.append(
            evidence
        )

        total_chars += len(
            evidence
        )

        sources.append(
            {
                "memory_id": memory_id_value,
                "memory_type": memory_type,
                "chunk_index": int(
                    chunk_index
                ),
                "distance": float(
                    distance
                ),
            }
        )

        if (
            strongest_distance is None
            or distance < strongest_distance
        ):
            strongest_distance = distance

    return (
        formatted_results,
        sources,
        strongest_distance,
    )


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


def _safe_calculate(
    node,
):
    """
    Recursively evaluate a restricted arithmetic AST.
    """

    if isinstance(
        node,
        ast.Constant,
    ):

        if isinstance(
            node.value,
            (int, float),
        ):
            return node.value

        raise ValueError(
            "Only numbers are allowed."
        )

    if isinstance(
        node,
        ast.UnaryOp,
    ):

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

    if isinstance(
        node,
        ast.BinOp,
    ):

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

        # Prevent dangerous exponentiation.
        if (
            isinstance(
                node.op,
                ast.Pow,
            )
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
    expression: str,
):

    if (
        not expression
        or not expression.strip()
    ):

        raise ValueError(
            "Expression cannot be empty."
        )

    try:

        tree = ast.parse(
            expression,
            mode="eval",
        )

        return _safe_calculate(
            tree.body
        )

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
    """

    if not isinstance(
        scope,
        dict,
    ):

        raise ValueError(
            "Invalid memory scope."
        )

    user_id = scope.get(
        "user_id"
    )

    if (
        not isinstance(
            user_id,
            str,
        )
        or not user_id.strip()
    ):

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

        if (
            not isinstance(
                memory_ids,
                list,
            )
            or not memory_ids
        ):

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

    if scope["scope_type"] == "all":
        return True

    return memory_id in (
        scope["memory_ids"] or []
    )


# ============================================================
# Create Tools
# ============================================================

def create_tools(
    scope: MemoryScope,
):

    _validate_scope(
        scope
    )

    user_id = scope[
        "user_id"
    ]

    # ========================================================
    # Tool 1 — Search Memories
    # ========================================================

    @tool(response_format="content_and_artifact")
    def search_memories(
        query: str,
    ):
        """
        Search the user's saved memories.
        """

        if (
            not query
            or not query.strip()
        ):

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
        # Determine selected-memory filter.
        # ----------------------------------------------------

        selected_memory_id = None

        if scope["scope_type"] == "memory":

            selected_memory_id = (
                scope["memory_ids"] or [None]
            )[0]

        # ----------------------------------------------------
        # Retrieve only the maximum number of chunks needed.
        # ----------------------------------------------------

        results = retrieve(
            question=query.strip(),
            user_id=user_id,
            top_k=MAX_MEMORY_SEARCH_CHUNKS,
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

        # ----------------------------------------------------
        # Grounding check.
        # ----------------------------------------------------

        grounded_results = (
            get_grounded_results(
                results
            )
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
                    "threshold":
                        DEFAULT_DISTANCE_THRESHOLD,
                },
            )

        # ----------------------------------------------------
        # Apply output limits AFTER grounding.
        #
        # This is important: do not change retrieval quality
        # merely because the model receives less text.
        # ----------------------------------------------------

        (
            formatted_results,
            sources,
            strongest_distance,
        ) = _build_memory_evidence(
            grounded_results,
            scope,
        )

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

        # ----------------------------------------------------
        # Confidence is calculated from the original grounded
        # retrieval score, not the truncated text.
        # ----------------------------------------------------

        if strongest_distance <= 0.4:

            confidence = (
                "High retrieval confidence"
            )

        else:

            confidence = (
                "Medium retrieval confidence"
            )

        artifact = {
            "grounded": True,
            "confidence": confidence,
            "distance": float(
                strongest_distance
            ),
            "threshold":
                DEFAULT_DISTANCE_THRESHOLD,
            "sources": sources,

            # Useful for Phase 0.2 observability/debugging.
            "limits": {
                "max_chunks":
                    MAX_MEMORY_SEARCH_CHUNKS,
                "max_chars_per_chunk":
                    MAX_CHARS_PER_MEMORY_CHUNK,
                "max_total_chars":
                    MAX_MEMORY_EVIDENCE_CHARS,
            },
        }

        return (
            "\n\n".join(
                formatted_results
            ),
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
        Get bounded details about one saved memory.
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
        # Scope security.
        # ----------------------------------------------------

        if not _is_memory_allowed(
            scope,
            memory_id_requested,
        ):

            return (
                "Memory not found."
            )

        # ----------------------------------------------------
        # User ownership is always enforced at DB level.
        # ----------------------------------------------------

        memory = get_memory_record(
            memory_id=memory_id_requested,
            user_id=user_id,
        )

        if memory is None:

            return (
                "Memory not found."
            )

        extracted_text = _truncate_text(
            memory.get(
                "extracted_text",
                "",
            ),
            MAX_MEMORY_DETAIL_CHARS,
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
{extracted_text}
""".strip()

    # ========================================================
    # Tool 3 — Calculator
    # ========================================================

    @tool
    def calculate(
        expression: str,
    ) -> str:
        """Calculate a mathematical expression."""

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
        """

        if not settings.TAVILY_API_KEY:

            return (
                "Web search is not configured. "
                "TAVILY_API_KEY is missing."
            )

        if (
            not query
            or not query.strip()
        ):

            return (
                "Web search query cannot be empty."
            )

        try:

            results = tavily_tool.invoke(
                {
                    "query": query.strip()
                }
            )

            # ------------------------------------------------
            # Hard cap on external tool output.
            # ------------------------------------------------

            return _truncate_text(
                str(results),
                MAX_WEB_RESULT_CHARS,
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