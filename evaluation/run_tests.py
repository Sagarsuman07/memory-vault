"""
Memory Vault - Evaluation Test Runner

Purpose:
    1. Read manually prepared golden_questions.xlsx
    2. Execute each test against the real Memory Vault agent
    3. Preserve the same LangGraph thread for the same conversation_id
    4. Capture answers, tools, retrieval sources, latency, scope, etc.
    5. Create:
         - raw_results.json
         - evaluation_results.xlsx

Usage:
    python evaluation/run_tests.py

Run from the Memory Vault project root.
"""

from __future__ import annotations

import json
import re
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Make project root importable when this file is executed as:
#
#     python evaluation/run_tests.py
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Existing Memory Vault imports
# ---------------------------------------------------------------------------

from agent.graph import run_langgraph_agent
from config.settings import settings


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EVALUATION_DIR = PROJECT_ROOT / "evaluation"

INPUT_EXCEL = EVALUATION_DIR / "golden_questions.xlsx"

RESULTS_DIR = EVALUATION_DIR / "results"


# ---------------------------------------------------------------------------
# Expected input columns
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "test_id",
    "conversation_id",
    "round",
    "scope_type",
    "memory_id",
    "query",
    "expected_answer",
    "expected_memory_ids",
    "expected_tools",
    "max_tool_calls",
]


# ---------------------------------------------------------------------------
# Manual evaluation columns
# ---------------------------------------------------------------------------

MANUAL_EVALUATION_COLUMNS = [
    "answer_correct",
    "retrieval_correct",
    "tool_correct",
    "tool_budget_ok",
    "human_correctness_1_5",
    "human_groundedness_1_5",
    "human_relevance_1_5",
    "human_hallucination_control_1_5",
    "human_scope_adherence_1_5",
    "human_tool_use_1_5",
    "human_conversation_context_1_5",
    "human_verdict",
    "human_comments",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def clean_value(value: Any) -> str:
    """
    Convert Excel values into clean strings.
    Empty/NaN values become an empty string.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def parse_integer(value: Any, field_name: str, row_number: int) -> int:
    """
    Parse an integer from Excel.
    """
    raw = clean_value(value)

    if not raw:
        raise ValueError(
            f"Row {row_number}: '{field_name}' cannot be empty."
        )

    try:
        return int(float(raw))
    except ValueError as exc:
        raise ValueError(
            f"Row {row_number}: '{field_name}' must be an integer. "
            f"Received: {raw!r}"
        ) from exc


def parse_list(value: Any) -> list[str]:
    """
    Parse comma/semicolon/newline separated values.

    Examples:
        search_memories
        search_memories,calculate
        memory_1;memory_2
    """
    raw = clean_value(value)

    if not raw:
        return []

    parts = re.split(r"[,;\n]+", raw)

    return [
        item.strip()
        for item in parts
        if item.strip()
    ]


def extract_tool_names(tool_calls: Any) -> list[str]:
    """
    Extract tool names from the result returned by run_langgraph_agent().

    Supports common dictionary formats and string-like fallback values.
    """
    if not tool_calls:
        return []

    names: list[str] = []

    for call in tool_calls:
        if isinstance(call, dict):
            name = (
                call.get("name")
                or call.get("tool")
                or call.get("tool_name")
            )

            if name:
                names.append(str(name))

        elif isinstance(call, str):
            names.append(call)

    return names


def extract_memory_ids(result: dict[str, Any]) -> list[str]:
    """
    Extract memory IDs from memory-level retrieval sources.
    """
    sources = result.get("retrieval_sources")

    if sources is None:
        sources = result.get("sources")

    if not sources:
        return []

    memory_ids: list[str] = []

    for source in sources:
        if not isinstance(source, dict):
            continue

        memory_id = source.get("memory_id")

        if memory_id:
            memory_ids.append(str(memory_id))

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(memory_ids))


def json_safe(value: Any) -> Any:
    """
    Convert common non-JSON-serializable objects into JSON-safe values.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item)
            for item in value
        ]

    return str(value)


def compare_memory_ids(
    expected_ids: list[str],
    actual_ids: list[str],
) -> str:
    """
    Check whether all expected memory IDs were retrieved.

    Extra retrieved memories are not automatically considered incorrect;
    manual retrieval evaluation can determine whether they were relevant.
    """
    if not expected_ids:
        return "NOT_APPLICABLE"

    expected_set = set(expected_ids)
    actual_set = set(actual_ids)

    if expected_set.issubset(actual_set):
        return "PASS"

    return "FAIL"


def compare_tools(
    expected_tools: list[str],
    actual_tools: list[str],
) -> str:
    """
    Compare expected and actual tool sets.

    Order is ignored.

    Example:
        expected = [search_memories]
        actual   = [search_memories]

        -> PASS
    """
    if not expected_tools:
        return "NOT_APPLICABLE"

    expected_set = set(expected_tools)
    actual_set = set(actual_tools)

    if expected_set == actual_set:
        return "PASS"

    return "FAIL"


def check_scope_isolation(
    scope_type: str,
    selected_memory_id: str,
    actual_memory_ids: list[str],
) -> str:
    """
    Basic automatic scope-isolation check.

    For memory scope:
        retrieved memories must belong to the selected memory.

    An empty retrieval is acceptable here because the query may legitimately
    have no matching information.
    """
    if scope_type != "memory":
        return "NOT_APPLICABLE"

    if not selected_memory_id:
        return "FAIL"

    if not actual_memory_ids:
        return "PASS"

    if all(
        memory_id == selected_memory_id
        for memory_id in actual_memory_ids
    ):
        return "PASS"

    return "FAIL"


def calculate_deterministic_pass(
    *,
    error: str,
    memory_check: str,
    tool_check: str,
    tool_budget_check: str,
    scope_isolation_check: str,
) -> str:
    """
    Calculate the overall automated result.

    Only objective automated checks are included here.

    Answer correctness is intentionally NOT evaluated automatically.
    It is evaluated manually using:
        - answer_correct
        - human_correctness_1_5

    NOT_APPLICABLE checks are ignored.
    """
    if error:
        return "FAIL"

    checks = [
        memory_check,
        tool_check,
        tool_budget_check,
        scope_isolation_check,
    ]

    applicable_checks = [
        check
        for check in checks
        if check != "NOT_APPLICABLE"
    ]

    if any(check == "FAIL" for check in applicable_checks):
        return "FAIL"

    return "PASS"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_input_dataframe(df: pd.DataFrame) -> None:
    """
    Validate the manually prepared golden dataset.
    """
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "golden_questions.xlsx is missing required columns:\n"
            + "\n".join(f"  - {column}" for column in missing_columns)
        )

    if df.empty:
        raise ValueError(
            "golden_questions.xlsx does not contain any test cases."
        )

    for index, row in df.iterrows():
        row_number = index + 2

        test_id = clean_value(row["test_id"])
        conversation_id = clean_value(row["conversation_id"])
        scope_type = clean_value(row["scope_type"]).lower()
        query = clean_value(row["query"])

        if not test_id:
            raise ValueError(
                f"Row {row_number}: test_id cannot be empty."
            )

        if not conversation_id:
            raise ValueError(
                f"Row {row_number}: conversation_id cannot be empty."
            )

        if scope_type not in {"all", "memory"}:
            raise ValueError(
                f"Row {row_number}: scope_type must be "
                f"'all' or 'memory'. Received: {scope_type!r}"
            )

        memory_id = clean_value(row["memory_id"])

        if scope_type == "memory" and not memory_id:
            raise ValueError(
                f"Row {row_number}: memory_id is required "
                f"when scope_type='memory'."
            )

        if scope_type == "all" and memory_id:
            raise ValueError(
                f"Row {row_number}: memory_id must be blank "
                f"when scope_type='all'."
            )

        if not query:
            raise ValueError(
                f"Row {row_number}: query cannot be empty."
            )

        parse_integer(
            row["round"],
            "round",
            row_number,
        )

        parse_integer(
            row["max_tool_calls"],
            "max_tool_calls",
            row_number,
        )


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

def run_tests() -> Path:
    """
    Main evaluation execution.
    """
    if not INPUT_EXCEL.exists():
        raise FileNotFoundError(
            f"Input Excel file not found:\n{INPUT_EXCEL}\n\n"
            "Create evaluation/golden_questions.xlsx first."
        )

    print("=" * 70)
    print("Memory Vault Evaluation")
    print("=" * 70)

    print(f"\nInput file:")
    print(f"  {INPUT_EXCEL}")

    # ---------------------------------------------------------------
    # Read Excel
    # ---------------------------------------------------------------

    df = pd.read_excel(INPUT_EXCEL)

    validate_input_dataframe(df)

    print(f"\nTotal test cases: {len(df)}")

    # ---------------------------------------------------------------
    # Create timestamped result directory
    # ---------------------------------------------------------------

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = RESULTS_DIR / timestamp
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    raw_results_path = output_dir / "raw_results.json"
    evaluation_excel_path = output_dir / "evaluation_results.xlsx"

    print(f"\nOutput directory:")
    print(f"  {output_dir}")

    # ---------------------------------------------------------------
    # Conversation thread mapping
    #
    # Same conversation_id -> same LangGraph thread.
    # Different conversation_id -> different thread.
    # ---------------------------------------------------------------

    conversation_threads: dict[str, str] = {}

    raw_results: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []

    # ---------------------------------------------------------------
    # Execute tests in Excel order.
    #
    # This is important because multi-round conversations depend
    # on previous rounds.
    # ---------------------------------------------------------------

    for index, row in df.iterrows():
        row_number = index + 2

        test_id = clean_value(row["test_id"])
        conversation_id = clean_value(row["conversation_id"])

        round_number = parse_integer(
            row["round"],
            "round",
            row_number,
        )

        scope_type = clean_value(
            row["scope_type"]
        ).lower()

        selected_memory_id = clean_value(
            row["memory_id"]
        )

        query = clean_value(row["query"])

        expected_answer = clean_value(
            row["expected_answer"]
        )

        expected_memory_ids = parse_list(
            row["expected_memory_ids"]
        )

        expected_tools = parse_list(
            row["expected_tools"]
        )

        max_tool_calls = parse_integer(
            row["max_tool_calls"],
            "max_tool_calls",
            row_number,
        )

        # -----------------------------------------------------------
        # Create/reuse LangGraph thread
        # -----------------------------------------------------------

        if conversation_id not in conversation_threads:
            conversation_threads[conversation_id] = (
                f"{settings.DEMO_USER_ID}:{uuid.uuid4()}"
            )

        thread_id = conversation_threads[conversation_id]

        print(
            f"\n[{index + 1}/{len(df)}] "
            f"{test_id} | "
            f"Conversation={conversation_id} | "
            f"Round={round_number}"
        )

        print(f"  Scope : {scope_type}")
        print(f"  Query : {query}")

        if selected_memory_id:
            print(f"  Memory: {selected_memory_id}")

        start_time = time.perf_counter()

        actual_answer = ""
        actual_memory_ids: list[str] = []
        actual_tools: list[str] = []

        tool_call_count = 0
        tool_limit_reached = False
        blocked_tool_calls: list[Any] = []

        actual_scope = ""
        actual_memory_id = ""

        retrieval_sources: list[Any] = []

        error = ""

        result: dict[str, Any] = {}

        # -----------------------------------------------------------
        # Execute actual Memory Vault agent
        # -----------------------------------------------------------

        try:
            result = run_langgraph_agent(
                question=query,
                user_id=settings.DEMO_USER_ID,
                thread_id=thread_id,
                memory_id=(
                    selected_memory_id
                    if scope_type == "memory"
                    else None
                ),
            )

            actual_answer = clean_value(
                result.get("answer")
            )

            actual_memory_ids = extract_memory_ids(result)

            actual_tools = extract_tool_names(
                result.get("tool_calls", [])
            )

            tool_call_count = int(
                result.get("tool_call_count", 0)
                or 0
            )

            tool_limit_reached = bool(
                result.get("tool_limit_reached", False)
            )

            blocked_tool_calls = (
                result.get("blocked_tool_calls", [])
                or []
            )

            actual_scope = clean_value(
                result.get("scope")
            )

            actual_memory_id = clean_value(
                result.get("memory_id")
            )

            retrieval_sources = (
                result.get("retrieval_sources")
                or result.get("sources")
                or []
            )

        except Exception as exc:
            error = (
                f"{type(exc).__name__}: {exc}"
            )

            print(f"  ERROR: {error}")

        latency_ms = round(
            (time.perf_counter() - start_time) * 1000,
            2,
        )

        # -----------------------------------------------------------
        # Automatic checks
        # -----------------------------------------------------------

        # Answer correctness is intentionally NOT checked automatically.
        # It is evaluated manually using answer_correct and the
        # human_correctness_1_5 score.

        memory_check = compare_memory_ids(
            expected_memory_ids,
            actual_memory_ids,
        )

        tool_check = compare_tools(
            expected_tools,
            actual_tools,
        )

        tool_budget_check = (
            "PASS"
            if tool_call_count <= max_tool_calls
            else "FAIL"
        )

        scope_isolation_check = check_scope_isolation(
            scope_type,
            selected_memory_id,
            actual_memory_ids,
        )

        deterministic_pass = calculate_deterministic_pass(
            error=error,
            memory_check=memory_check,
            tool_check=tool_check,
            tool_budget_check=tool_budget_check,
            scope_isolation_check=scope_isolation_check,
        )

        # -----------------------------------------------------------
        # Raw result
        # -----------------------------------------------------------

        raw_case = {
            "test_id": test_id,
            "conversation_id": conversation_id,
            "round": round_number,
            "thread_id": thread_id,
            "input": {
                "scope_type": scope_type,
                "memory_id": selected_memory_id,
                "query": query,
                "expected_answer": expected_answer,
                "expected_memory_ids": expected_memory_ids,
                "expected_tools": expected_tools,
                "max_tool_calls": max_tool_calls,
            },
            "execution": {
                "actual_answer": actual_answer,
                "actual_memory_ids": actual_memory_ids,
                "actual_tools": actual_tools,
                "tool_call_count": tool_call_count,
                "tool_limit_reached": tool_limit_reached,
                "blocked_tool_calls": json_safe(
                    blocked_tool_calls
                ),
                "scope": actual_scope,
                "memory_id": actual_memory_id,
                "retrieval_sources": json_safe(
                    retrieval_sources
                ),
                "latency_ms": latency_ms,
                "error": error,
            },
            "checks": {
                "memory_check": memory_check,
                "tool_check": tool_check,
                "tool_budget_check": tool_budget_check,
                "scope_isolation_check": scope_isolation_check,
                "deterministic_pass": deterministic_pass,
            },
            "raw_agent_result": json_safe(result),
        }

        raw_results.append(raw_case)

        # -----------------------------------------------------------
        # Evaluation Excel row
        # -----------------------------------------------------------

        evaluation_row = {
            # Original input columns
            "test_id": test_id,
            "conversation_id": conversation_id,
            "round": round_number,
            "scope_type": scope_type,
            "memory_id": selected_memory_id,
            "query": query,
            "expected_answer": expected_answer,
            "expected_memory_ids": ", ".join(
                expected_memory_ids
            ),
            "expected_tools": ", ".join(
                expected_tools
            ),
            "max_tool_calls": max_tool_calls,

            # Automatic execution results
            "actual_answer": actual_answer,
            "actual_memory_ids": ", ".join(
                actual_memory_ids
            ),
            "actual_tools": ", ".join(
                actual_tools
            ),
            "tool_call_count": tool_call_count,
            "tool_limit_reached": tool_limit_reached,
            "blocked_tool_calls": json.dumps(
                json_safe(blocked_tool_calls),
                ensure_ascii=False,
            ),
            "actual_scope": actual_scope,
            "actual_memory_id": actual_memory_id,
            "retrieval_sources": json.dumps(
                json_safe(retrieval_sources),
                ensure_ascii=False,
            ),
            "thread_id": thread_id,
            "latency_ms": latency_ms,
            "error": error,

            # Automatic checks
            "memory_check": memory_check,
            "tool_check": tool_check,
            "tool_budget_check": tool_budget_check,
            "scope_isolation_check": scope_isolation_check,
            "deterministic_pass": deterministic_pass,

            # Manual evaluation columns
            "answer_correct": "",
            "retrieval_correct": "",
            "tool_correct": "",
            "tool_budget_ok": "",
            "human_correctness_1_5": "",
            "human_groundedness_1_5": "",
            "human_relevance_1_5": "",
            "human_hallucination_control_1_5": "",
            "human_scope_adherence_1_5": "",
            "human_tool_use_1_5": "",
            "human_conversation_context_1_5": "",
            "human_verdict": "",
            "human_comments": "",
        }

        evaluation_rows.append(
            evaluation_row
        )

        # -----------------------------------------------------------
        # Console summary
        # -----------------------------------------------------------

        print(f"  Answer : {actual_answer}")
        print(f"  Tools  : {actual_tools}")
        print(f"  Calls  : {tool_call_count}")
        print(f"  Memory : {actual_memory_ids}")
        print(f"  Time   : {latency_ms} ms")
        print(
            f"  Result : {deterministic_pass}"
        )

    # ----------------------------------------------------------------
    # Save raw JSON
    # ----------------------------------------------------------------

    raw_payload = {
        "evaluation": {
            "timestamp": timestamp,
            "user_id": settings.DEMO_USER_ID,
            "input_file": str(INPUT_EXCEL),
            "total_tests": len(raw_results),
        },
        "results": raw_results,
    }

    with raw_results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            raw_payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    # ----------------------------------------------------------------
    # Save evaluation Excel
    # ----------------------------------------------------------------

    evaluation_df = pd.DataFrame(
        evaluation_rows
    )

    evaluation_df.to_excel(
        evaluation_excel_path,
        index=False,
        sheet_name="Evaluation Results",
    )

    # ----------------------------------------------------------------
    # Final console output
    # ----------------------------------------------------------------

    print("\n" + "=" * 70)
    print("Evaluation execution completed.")
    print("=" * 70)

    print(f"\nRaw results:")
    print(f"  {raw_results_path}")

    print(f"\nEvaluation Excel:")
    print(f"  {evaluation_excel_path}")

    print(
        "\nIMPORTANT:"
        "\nOpen evaluation_results.xlsx and manually fill the "
        "human evaluation columns before generating the report."
    )

    return evaluation_excel_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_tests()