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

MAX_MEMORY_SEARCH_CHUNKS = 5

MAX_CHARS_PER_MEMORY_CHUNK = 1500

MAX_MEMORY_EVIDENCE_CHARS = 7500

MAX_MEMORY_DETAIL_CHARS = 4000

MAX_WEB_RESULT_CHARS = 6000

TRUNCATION_MARKER = "\n...[truncated]"


# ============================================================
# Generic Text Limiter
# ============================================================

def _truncate_text(
    text: str,
    max_chars: int,
) -> str:

    if not text:
        return ""

    if max_chars <= 0:
        return ""

    text = str(text).strip()

    if len(text) <= max_chars:
        return text

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
# Memory Evidence Builder
# ============================================================

def _resolve_memory_title(
    document,
    memory_id: str,
    scope: MemoryScope,
) -> str:
    """
    Resolve the memory title.

    First tries vector metadata.
    Falls back to the database if the title is unavailable.
    """

    title = document.metadata.get(
        "title"
    )

    if not title:

        try:

            memory = get_memory_record(
                memory_id=memory_id,
                user_id=scope["user_id"],
            )

            if memory:
                title = memory.get(
                    "title"
                )

        except Exception:

            title = None

    return _truncate_text(
        title or "Untitled Memory",
        120,
    )


def _build_memory_evidence(
    grounded_results,
    scope: MemoryScope,
):
    """
    Build bounded evidence for the LLM.

    Evidence is chunk-level.

    Sources are memory-level.

    Therefore, if multiple chunks belong to the same memory,
    the LLM receives the relevant chunks while the UI receives
    only one source entry for that memory.
    """

    formatted_results = []

    # Memory-level sources.
    sources_by_memory = {}

    # Titles already resolved in this call.
    titles = {}

    total_chars = 0

    strongest_distance = None

    evidence_rank = 0

    for document, distance in grounded_results:

        # ----------------------------------------------------
        # Chunk limit
        # ----------------------------------------------------

        if (
            evidence_rank
            >= MAX_MEMORY_SEARCH_CHUNKS
        ):
            break

        memory_id = document.metadata.get(
            "memory_id"
        )

        if not memory_id:
            continue

        # ----------------------------------------------------
        # Authorization check
        # ----------------------------------------------------

        if not _is_memory_allowed(
            scope,
            memory_id,
        ):
            continue

        memory_type = document.metadata.get(
            "memory_type",
            "unknown",
        )

        # ----------------------------------------------------
        # Character budget
        # ----------------------------------------------------

        remaining_chars = (
            MAX_MEMORY_EVIDENCE_CHARS
            - total_chars
        )

        if remaining_chars <= 0:
            break

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

        # The final LLM needs to see the memory ID
        # so it can report USED_MEMORY_IDS.
        if memory_id not in titles:

            titles[memory_id] = _resolve_memory_title(
                document,
                memory_id,
                scope,
            )

        title = titles[memory_id]

        evidence = (
            f"Evidence {evidence_rank}\n"
            f"Memory ID: {memory_id}\n"
            f"Title: {title}\n\n"
            f"Content:\n"
            f"{content}"
        )

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

        # ----------------------------------------------------
        # Memory-level source
        # ----------------------------------------------------

        if memory_id not in sources_by_memory:

            sources_by_memory[
                memory_id
            ] = {
                "memory_id": memory_id,
                "memory_type": memory_type,
                "title": (
                    title
                    or "Untitled Memory"
                ),
                "distance": float(
                    distance
                ),
            }

        else:

            # Keep strongest distance
            # for this memory.
            current_distance = (
                sources_by_memory[
                    memory_id
                ]["distance"]
            )

            if distance < current_distance:

                sources_by_memory[
                    memory_id
                ]["distance"] = float(
                    distance
                )

        # ----------------------------------------------------
        # Overall strongest retrieval
        # ----------------------------------------------------

        if (
            strongest_distance is None
            or distance < strongest_distance
        ):

            strongest_distance = distance

    return (
        formatted_results,
        list(
            sources_by_memory.values()
        ),
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


def _safe_calculate(node):

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

    @tool(
        response_format="content_and_artifact"
    )
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
        # Selected memory
        # ----------------------------------------------------

        selected_memory_id = None

        if (
            scope["scope_type"]
            == "memory"
        ):

            selected_memory_id = (
                scope["memory_ids"]
                or [None]
            )[0]

        # ----------------------------------------------------
        # Retrieval
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
        # Grounding
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
        # Build evidence + sources
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
        # Confidence
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
    # Tool 2 — Calculator
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
    # Tool 3 — Web Search
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

            return _truncate_text(
                str(results),
                MAX_WEB_RESULT_CHARS,
            )

        except Exception as error:

            return (
                f"Web search failed: {error}"
            )

    # ========================================================
    # Return ONLY tools that the agent should be allowed to use
    # ========================================================

    return [
        search_memories,
        calculate,
        search_web,
    ]