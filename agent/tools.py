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
    node
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
# Create Tools
# ============================================================

def create_tools(
    user_id: str,
    memory_id: str | None = None,
):

    if not user_id or not user_id.strip():
        raise ValueError(
            "User ID cannot be empty."
        )

    if memory_id is not None and not memory_id.strip():
        raise ValueError(
            "Memory ID cannot be empty."
        )

    # memory_id is application-controlled.
    # The LLM never chooses the retrieval scope.


    # ========================================================
    # Tool 1 — Search Memories
    # ========================================================

    @tool
    def search_memories(
        query: str,
    ) -> str:
        """
        Search the user's saved memories using semantic search.

        The search is automatically restricted to the current
        chat scope. The model cannot choose user_id or memory_id.
        """

        if not query or not query.strip():

            return (
                "No search query was provided."
            )


        results = retrieve(
            question=query,
            user_id=user_id,
            top_k=5,
            memory_id=memory_id,
        )


        if not results:

            return (
                "No relevant memories were found."
            )


        formatted_results = []


        for rank, (
            document,
            distance,
        ) in enumerate(
            results,
            start=1,
        ):

            formatted_results.append(
                f"""
Result {rank}

Memory ID:
{document.metadata["memory_id"]}

Memory Type:
{document.metadata["memory_type"]}

Chunk Index:
{document.metadata["chunk_index"]}

Distance:
{distance}

Content:
{document.page_content}
"""
            )


        return "\n\n".join(
            formatted_results
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

        In a memory-scoped chat, this operation is restricted
        to the selected memory.
        """

        if (
            not memory_id_requested
            or not memory_id_requested.strip()
        ):

            return (
                "Memory ID cannot be empty."
            )


        # ----------------------------------------------------
        # Scope security
        # ----------------------------------------------------

        if (
            memory_id is not None
            and memory_id_requested != memory_id
        ):

            return (
                "Memory not found."
            )


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

        Use this tool only when the requested information
        is not expected to come from the user's memories,
        or when the user explicitly asks for current
        or web information.
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