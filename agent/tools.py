import ast
import operator as op

from langchain_core.tools import tool
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
):

    # ========================================================
    # Tool 1 — Search Memories
    # ========================================================

    @tool
    def search_memories(
        query: str,
    ) -> str:
        """
        Search the user's saved memories using semantic search.

        Use this tool when the user asks about information
        that may exist in their saved memories.
        """

        if not query or not query.strip():

            return (
                "No search query was provided."
            )


        results = retrieve(
            question=query,
            user_id=user_id,
            top_k=5,
            memory_id=None,
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
        memory_id: str,
    ) -> str:
        """
        Get detailed information about one saved memory.

        Use this tool when a specific memory ID is known
        and more information about that memory is required.
        """

        if not memory_id or not memory_id.strip():

            return (
                "Memory ID cannot be empty."
            )


        memory = get_memory_record(
            memory_id=memory_id,
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




from langchain_groq import ChatGroq
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)


# ============================================================
# Tool Calling
# ============================================================

def run_tool_calling(
    question: str,
    user_id: str,
    max_iterations: int = 5,
):

    if not question or not question.strip():

        raise ValueError(
            "Question cannot be empty."
        )


    tools = create_tools(
        user_id
    )


    tool_map = {
        tool.name: tool
        for tool in tools
    }


    # ========================================================
    # Create LLM
    # ========================================================

    if not settings.GROQ_API_KEY:

        raise ValueError(
            "GROQ_API_KEY is not configured."
        )


    model = ChatGroq(
        model=settings.GROQ_LLM_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY,
    )


    model_with_tools = model.bind_tools(
        tools
    )


    # ========================================================
    # Initial messages
    # ========================================================

    messages = [
        SystemMessage(
            content="""
You are Memory Vault's tool-calling assistant.

You have access to four tools:

1. search_memories
   Search the user's saved memories.

2. get_memory
   Retrieve a specific saved memory.

3. calculate
   Perform arithmetic calculations.

4. search_web
   Search the internet for current or external information.

Rules:

- Use search_memories when information may exist
  in the user's saved memories.
- Use get_memory when you need detailed information
  about a specific memory.
- Use calculate when arithmetic is required.
- Use search_web only when outside/current information
  is needed or the user explicitly asks for web information.
- Do not invent memory information.
- Do not claim that a tool returned information
  that it did not return.
- After receiving tool results, answer the user directly.
""",
        ),
        HumanMessage(
            content=question
        ),
    ]


    tool_trace = []


    # ========================================================
    # Tool-calling loop
    # ========================================================

    for _ in range(
        max_iterations
    ):

        response = model_with_tools.invoke(
            messages
        )


        messages.append(
            response
        )


        # ====================================================
        # No tool call → final answer
        # ====================================================

        if not response.tool_calls:

            return {
                "answer": response.content,
                "tool_calls": tool_trace,
            }


        # ====================================================
        # Execute tool calls
        # ====================================================

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]

            tool_args = tool_call["args"]


            tool_trace.append(
                {
                    "tool": tool_name,
                    "arguments": tool_args,
                }
            )


            selected_tool = tool_map.get(
                tool_name
            )


            if selected_tool is None:

                raise ValueError(
                    f"Unknown tool requested: "
                    f"{tool_name}"
                )


            tool_result = selected_tool.invoke(
                tool_call
            )


            messages.append(
                tool_result
            )


    raise RuntimeError(
        "Tool-calling loop exceeded the maximum "
        "number of iterations."
    )