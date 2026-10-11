Yes. I agree with this redesign. The previous implementation was becoming too much like a testing framework. For your Memory Vault project, a **simple, understandable evaluation pipeline** is better.

I would reduce it to exactly **two Python files**.

## Final simplified design

```text
memory-vault/
│
├── memory_vault.db
├── app.py
├── database/
├── agent/
├── rag/
├── ...
│
└── evaluation/
    ├── run_tests.py
    └── generate_report.py
```

You manually create only **one input Excel**:

```text
evaluation/
└── golden_questions.xlsx
```

No `export_memories.py`, no question-template generator, no separate baseline script, no security-regression script.

---

# 1. Your input Excel

You prepare this manually:

```text
golden_questions.xlsx
```

with exactly these columns:

| Column                | Purpose                                             |
| --------------------- | --------------------------------------------------- |
| `test_id`             | Unique test ID                                      |
| `conversation_id`     | Same ID for questions belonging to one conversation |
| `round`               | Conversation round: 1, 2, 3...                      |
| `scope_type`          | `all` or `memory`                                   |
| `memory_id`           | Required when scope is `memory`; blank for `all`    |
| `query`               | Actual question                                     |
| `expected_answer`     | Expected answer/key information                     |
| `expected_memory_ids` | Expected memory IDs when scope is `all`             |
| `expected_tools`      | Tool(s) expected to be used                         |
| `max_tool_calls`      | Maximum allowed tool calls                          |

### Example

| test_id | conversation_id | round | scope_type | memory_id | query                        | expected_answer | expected_memory_ids | expected_tools  | max_tool_calls |
| ------- | --------------- | ----: | ---------- | --------- | ---------------------------- | --------------- | ------------------- | --------------- | -------------: |
| T001    | C001            |     1 | memory     | `abc123`  | What was my headphone price? | ₹25,000         |                     | search_memories |              4 |
| T002    | C001            |     2 | memory     | `abc123`  | When did I buy them?         | September 2026  |                     | search_memories |              4 |
| T003    | C002            |     1 | all        |           | What was my hotel name?      | Hotel ABC       | `hotel_id`          | search_memories |              4 |
| T004    | C003            |     1 | memory     | `abc123`  | Calculate 25 + 75            | 100             |                     | calculate       |              4 |

For a multi-round conversation:

```text
C001 → Round 1
C001 → Round 2
C001 → Round 3
```

The test runner will use the **same LangGraph thread** for all three.

---

# 2. Python file #1 — `run_tests.py`

This does only one job:

```text
golden_questions.xlsx
        ↓
read rows
        ↓
group by conversation
        ↓
run queries
        ↓
capture actual results
        ↓
create timestamp folder
        ↓
raw_results.json
evaluation_results.xlsx
```

The evaluation Excel will contain your original input columns plus the actual execution/evaluation columns.

I recommend these additional columns:

```text
actual_answer
actual_memory_ids
actual_tools
tool_call_count
tool_limit_reached
blocked_tool_calls
scope
latency_ms

answer_correct
retrieval_correct
tool_correct
tool_budget_ok
human_correctness_1_5
human_groundedness_1_5
human_relevance_1_5
human_hallucination_control_1_5
human_scope_adherence_1_5
human_tool_use_1_5
human_conversation_context_1_5
human_verdict
human_comments
```

The first group is **automatically populated**.

The `human_*` columns remain blank for you to evaluate manually.

---

# 3. Output structure

Every execution creates a new folder automatically:

```text
evaluation/
│
├── golden_questions.xlsx
│
├── run_tests.py
├── generate_report.py
│
└── results/
    │
    └── 2026-10-11_02-57-31/
        │
        ├── raw_results.json
        └── evaluation_results.xlsx
```

So you never overwrite an old evaluation run.

For example:

```text
results/
├── 2026-10-11_02-57-31/
│   ├── raw_results.json
│   └── evaluation_results.xlsx
│
├── 2026-10-15_18-22-10/
│   ├── raw_results.json
│   └── evaluation_results.xlsx
│
└── 2026-10-20_11-05-43/
    ├── raw_results.json
    └── evaluation_results.xlsx
```

This also gives you a very clean history of your evaluation runs.

---

# 4. Python file #2 — `generate_report.py`

After you manually evaluate:

```text
evaluation_results.xlsx
```

you run:

```bash
python evaluation/generate_report.py <evaluation_results.xlsx>
```

It will create:

```text
evaluation_report.xlsx
```

in the **same timestamp folder**.

So the final folder becomes:

```text
2026-10-11_02-57-31/
│
├── raw_results.json
├── evaluation_results.xlsx
└── evaluation_report.xlsx
```

---

# What should be inside the report?

Keep this simple too.

### Sheet 1 — Summary

Something like:

| Metric                        |  Result |
| ----------------------------- | ------: |
| Total Test Cases              |      25 |
| Passed Cases                  |      22 |
| Failed Cases                  |       3 |
| Deterministic Pass Rate       |     88% |
| Average Correctness           | 4.4 / 5 |
| Average Groundedness          | 4.5 / 5 |
| Average Relevance             | 4.3 / 5 |
| Average Hallucination Control | 4.6 / 5 |
| Average Scope Adherence       | 4.8 / 5 |
| Average Tool Use              | 4.4 / 5 |
| Average Conversation Context  | 4.2 / 5 |
| Average Tool Calls            |     1.8 |
| Tool Budget Violations        |       0 |

### Sheet 2 — Test Results

One row per test:

```text
test_id
conversation_id
round
scope_type
query
expected_answer
actual_answer
expected_memory_ids
actual_memory_ids
expected_tools
actual_tools
tool_call_count
answer_correct
retrieval_correct
tool_correct
tool_budget_ok
human scores
human verdict
comments
```

### Sheet 3 — Failed Cases

Only failed tests.

This makes it easy to identify what needs improvement.

---

# One important design decision

I recommend **not putting security/regression logic into these two scripts**.

Instead, security cases are simply normal rows in your Excel.

For example:

```text
test_id: SEC-001
scope_type: memory
memory_id: headphone_memory_id
query: What was my hotel name?
expected_answer: Information should not be found
expected_memory_ids:
expected_tools: search_memories
max_tool_calls: 4
```

Then you manually evaluate whether the model correctly respected the scope.

This is much easier to understand and explain in an interview:

> "I maintain a human-authored golden dataset containing normal, conversational, not-found, tool-use, and scope-isolation cases. The evaluation runner executes the same dataset against the real agent and records retrieval, tool usage, latency, and answers. I then manually score the outputs and generate an evaluation report."

That's a **much cleaner portfolio story**.

---

## I recommend this exact architecture

```text
                    ┌─────────────────────┐
                    │ golden_questions.xlsx│
                    │  Human authored      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    run_tests.py     │
                    │                     │
                    │ Read Excel          │
                    │ Run Agent           │
                    │ Preserve threads    │
                    │ Capture results     │
                    │ Calculate checks    │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
        raw_results.json            evaluation_results.xlsx
                                             │
                                             │
                                      Manual evaluation
                                             │
                                             ▼
                                   ┌──────────────────┐
                                   │ generate_report  │
                                   │      .py         │
                                   └────────┬─────────┘
                                            │
                                            ▼
                                   evaluation_report.xlsx
```

This is the version I would recommend for **your Memory Vault V1**.

And importantly, it still works with your existing architecture: `run_langgraph_agent()` already accepts `question`, `user_id`, `thread_id`, and optional `memory_id`, and returns the answer, scope, retrieval sources, tool calls, tool count, and blocked calls that we need.  

Your backend-controlled scope also remains untouched; the test runner simply passes the selected `memory_id` to the existing graph rather than trying to implement authorization itself. 

**I would use this simplified design rather than the previous five-script design.**















Yes. Based on your uploaded implementation design, I recommend implementing the evaluation system with **exactly two Python files**: `run_tests.py` and `generate_report.py`. The runner will reuse your existing `run_langgraph_agent()` and preserve the same LangGraph thread for multi-round conversations. 

# 1. Final folder structure

Add only this `evaluation` folder to your existing Memory Vault project:

```text
memory-vault/
│
├── app.py
├── memory_vault.db
│
├── config/
├── database/
├── ingestion/
├── memory/
├── rag/
├── agent/
│
└── evaluation/
    │
    ├── golden_questions.xlsx
    ├── run_tests.py
    ├── generate_report.py
    │
    └── results/
        └── 2026-10-11_02-57-31/
            ├── raw_results.json
            ├── evaluation_results.xlsx
            └── evaluation_report.xlsx
```

The `results/<timestamp>/` folder is created automatically by `run_tests.py`. This follows the design in your implementation document. 

---

# 2. Install required package

You need `pandas` and `openpyxl` for Excel handling.

From your project root:

```bash
pip install pandas openpyxl
```

Your existing Memory Vault dependencies remain unchanged.

---

# 3. Create `golden_questions.xlsx`

Create:

```text
evaluation/golden_questions.xlsx
```

Use **exactly these input columns**:

| Column                | Required                                            |
| --------------------- | --------------------------------------------------- |
| `test_id`             | Yes                                                 |
| `conversation_id`     | Yes                                                 |
| `round`               | Yes                                                 |
| `scope_type`          | Yes                                                 |
| `memory_id`           | Required for `memory` scope                         |
| `query`               | Yes                                                 |
| `expected_answer`     | Yes                                                 |
| `expected_memory_ids` | Required for `all` scope when retrieval is expected |
| `expected_tools`      | Yes                                                 |
| `max_tool_calls`      | Yes                                                 |

This matches your selected design. 

### Example

| test_id | conversation_id | round | scope_type | memory_id | query                        | expected_answer | expected_memory_ids | expected_tools  | max_tool_calls |
| ------- | --------------- | ----: | ---------- | --------- | ---------------------------- | --------------- | ------------------- | --------------- | -------------: |
| T001    | C001            |     1 | memory     | abc123    | What was my headphone price? | ₹25,000         |                     | search_memories |              4 |
| T002    | C001            |     2 | memory     | abc123    | When did I buy them?         | September 2026  |                     | search_memories |              4 |
| T003    | C002            |     1 | all        |           | What was my hotel name?      | Hotel ABC       | hotel_id            | search_memories |              4 |
| T004    | C003            |     1 | all        |           | Calculate 25 + 75            | 100             |                     | calculate       |              4 |

For conversation `C001`, the runner will use the **same thread** for rounds 1 and 2. 

---

# 4. `evaluation/run_tests.py`

Create this complete file:

```python
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
from config import settings


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


def normalize_text(value: Any) -> str:
    """
    Normalize text for simple deterministic comparison.

    This is intentionally NOT semantic evaluation.
    Human evaluation remains the final correctness decision.
    """
    text = clean_value(value).lower()

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


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


def compare_expected_answer(
    expected_answer: str,
    actual_answer: str,
) -> str:
    """
    Simple deterministic answer check.

    IMPORTANT:
    This is only a quick automated signal.
    It should NOT replace manual correctness evaluation.
    """
    expected = normalize_text(expected_answer)
    actual = normalize_text(actual_answer)

    if not expected:
        return "NOT_APPLICABLE"

    if not actual:
        return "FAIL"

    if expected in actual:
        return "PASS"

    return "FAIL"


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
    answer_check: str,
    memory_check: str,
    tool_check: str,
    tool_budget_check: str,
    scope_isolation_check: str,
) -> str:
    """
    Calculate an overall automated result.

    NOT_APPLICABLE checks are ignored.

    Human evaluation remains the authoritative final evaluation.
    """
    if error:
        return "FAIL"

    checks = [
        answer_check,
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

        answer_check = compare_expected_answer(
            expected_answer,
            actual_answer,
        )

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
            answer_check=answer_check,
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
                "answer_check": answer_check,
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
            "answer_check": answer_check,
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
```

---

# 5. What `run_tests.py` does

The flow is intentionally simple:

```text
golden_questions.xlsx
        │
        ▼
   run_tests.py
        │
        ├── Read Excel
        │
        ├── Create thread for conversation
        │
        ├── Reuse thread for next round
        │
        ├── Call run_langgraph_agent()
        │
        ├── Capture answer
        ├── Capture tools
        ├── Capture memory IDs
        ├── Capture tool count
        ├── Capture scope
        ├── Capture latency
        │
        ├── Run simple automatic checks
        │
        └── Create timestamp folder
                 │
                 ├── raw_results.json
                 └── evaluation_results.xlsx
```

This directly follows the intended runner behavior in your design. 

---

# 6. Important behavior for conversations

Suppose your Excel contains:

```text
C001 | Round 1 | What was my laptop price?
C001 | Round 2 | When did I purchase it?
C001 | Round 3 | What was the store name?
```

The runner creates:

```text
demo_user:<uuid>
```

for `C001`.

All three queries use that same thread:

```text
C001
 │
 ├── Round 1 ──┐
 ├── Round 2 ──┼── same LangGraph thread
 └── Round 3 ──┘
```

But:

```text
C002
```

gets another thread.

This is important because your evaluation should test the **real conversational behavior**, not three independent questions.

---

# 7. `evaluation/generate_report.py`

After manually evaluating `evaluation_results.xlsx`, create this second file:

```python
"""
Memory Vault - Evaluation Report Generator

Purpose:
    Read a manually evaluated evaluation_results.xlsx and create:

        evaluation_report.xlsx

The report contains:
    1. Summary
    2. Test Results
    3. Failed Cases

Usage:

    python evaluation/generate_report.py \
        evaluation/results/2026-10-11_02-57-31/evaluation_results.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANUAL_SCORE_COLUMNS = [
    "human_correctness_1_5",
    "human_groundedness_1_5",
    "human_relevance_1_5",
    "human_hallucination_control_1_5",
    "human_scope_adherence_1_5",
    "human_tool_use_1_5",
    "human_conversation_context_1_5",
]

PASS_FAIL_COLUMNS = [
    "answer_correct",
    "retrieval_correct",
    "tool_correct",
    "tool_budget_ok",
]


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_status(value: Any) -> str:
    """
    Normalize manual PASS/FAIL values.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip().upper()


def numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Convert a column to numeric values.
    Invalid/blank values become NaN.
    """
    if column not in df.columns:
        return pd.Series(
            dtype="float64"
        )

    return pd.to_numeric(
        df[column],
        errors="coerce",
    )


def average_score(
    df: pd.DataFrame,
    column: str,
) -> float | None:
    """
    Calculate average 1-5 manual score.
    """
    series = numeric_series(
        df,
        column,
    ).dropna()

    if series.empty:
        return None

    return round(
        float(series.mean()),
        2,
    )


def percentage(
    numerator: int,
    denominator: int,
) -> float:
    """
    Percentage helper.
    """
    if denominator == 0:
        return 0.0

    return round(
        (numerator / denominator) * 100,
        2,
    )


def count_status(
    df: pd.DataFrame,
    column: str,
    status: str,
) -> int:
    """
    Count PASS/FAIL values in a column.
    """
    if column not in df.columns:
        return 0

    values = df[column].map(
        normalize_status
    )

    return int(
        (values == status).sum()
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    evaluation_path: Path,
) -> Path:

    if not evaluation_path.exists():
        raise FileNotFoundError(
            f"Evaluation file not found:\n{evaluation_path}"
        )

    if evaluation_path.suffix.lower() != ".xlsx":
        raise ValueError(
            "The input file must be an .xlsx file."
        )

    print("=" * 70)
    print("Memory Vault Evaluation Report")
    print("=" * 70)

    print(
        f"\nReading:\n  {evaluation_path}"
    )

    df = pd.read_excel(
        evaluation_path,
        sheet_name=0,
    )

    if df.empty:
        raise ValueError(
            "The evaluation Excel file contains no test cases."
        )

    total_tests = len(df)

    # ----------------------------------------------------------------
    # Deterministic results
    # ----------------------------------------------------------------

    deterministic_pass_count = count_status(
        df,
        "deterministic_pass",
        "PASS",
    )

    deterministic_fail_count = count_status(
        df,
        "deterministic_pass",
        "FAIL",
    )

    deterministic_pass_rate = percentage(
        deterministic_pass_count,
        total_tests,
    )

    # ----------------------------------------------------------------
    # Manual verdict
    # ----------------------------------------------------------------

    manual_pass_count = count_status(
        df,
        "human_verdict",
        "PASS",
    )

    manual_fail_count = count_status(
        df,
        "human_verdict",
        "FAIL",
    )

    manual_evaluated_count = (
        manual_pass_count
        + manual_fail_count
    )

    manual_pass_rate = percentage(
        manual_pass_count,
        manual_evaluated_count,
    )

    # ----------------------------------------------------------------
    # Automatic check rates
    # ----------------------------------------------------------------

    answer_pass = count_status(
        df,
        "answer_correct",
        "PASS",
    )

    retrieval_pass = count_status(
        df,
        "retrieval_correct",
        "PASS",
    )

    tool_pass = count_status(
        df,
        "tool_correct",
        "PASS",
    )

    budget_pass = count_status(
        df,
        "tool_budget_ok",
        "PASS",
    )

    # ----------------------------------------------------------------
    # Tool calls
    # ----------------------------------------------------------------

    tool_calls = numeric_series(
        df,
        "tool_call_count",
    ).dropna()

    average_tool_calls = (
        round(
            float(tool_calls.mean()),
            2,
        )
        if not tool_calls.empty
        else None
    )

    tool_budget_violations = count_status(
        df,
        "tool_budget_check",
        "FAIL",
    )

    # ----------------------------------------------------------------
    # Latency
    # ----------------------------------------------------------------

    latency = numeric_series(
        df,
        "latency_ms",
    ).dropna()

    average_latency_ms = (
        round(
            float(latency.mean()),
            2,
        )
        if not latency.empty
        else None
    )

    # ----------------------------------------------------------------
    # Human score averages
    # ----------------------------------------------------------------

    score_averages = {
        column: average_score(
            df,
            column,
        )
        for column in MANUAL_SCORE_COLUMNS
    }

    # ----------------------------------------------------------------
    # Summary data
    # ----------------------------------------------------------------

    summary_rows = [
        {
            "Metric": "Total Test Cases",
            "Result": total_tests,
        },
        {
            "Metric": "Deterministic Passed Cases",
            "Result": deterministic_pass_count,
        },
        {
            "Metric": "Deterministic Failed Cases",
            "Result": deterministic_fail_count,
        },
        {
            "Metric": "Deterministic Pass Rate",
            "Result": f"{deterministic_pass_rate}%",
        },
        {
            "Metric": "Human Evaluated Cases",
            "Result": manual_evaluated_count,
        },
        {
            "Metric": "Human Passed Cases",
            "Result": manual_pass_count,
        },
        {
            "Metric": "Human Failed Cases",
            "Result": manual_fail_count,
        },
        {
            "Metric": "Human Pass Rate",
            "Result": (
                f"{manual_pass_rate}%"
                if manual_evaluated_count
                else "N/A"
            ),
        },
        {
            "Metric": "Answer Correct - PASS",
            "Result": answer_pass,
        },
        {
            "Metric": "Retrieval Correct - PASS",
            "Result": retrieval_pass,
        },
        {
            "Metric": "Tool Correct - PASS",
            "Result": tool_pass,
        },
        {
            "Metric": "Tool Budget - PASS",
            "Result": budget_pass,
        },
        {
            "Metric": "Tool Budget Violations",
            "Result": tool_budget_violations,
        },
        {
            "Metric": "Average Tool Calls",
            "Result": (
                average_tool_calls
                if average_tool_calls is not None
                else "N/A"
            ),
        },
        {
            "Metric": "Average Latency (ms)",
            "Result": (
                average_latency_ms
                if average_latency_ms is not None
                else "N/A"
            ),
        },
        {
            "Metric": "Average Correctness (1-5)",
            "Result": (
                score_averages[
                    "human_correctness_1_5"
                ]
                if score_averages[
                    "human_correctness_1_5"
                ] is not None
                else "N/A"
            ),
        },
        {
            "Metric": "Average Groundedness (1-5)",
            "Result": (
                score_averages[
                    "human_groundedness_1_5"
                ]
                if score_averages[
                    "human_groundedness_1_5"
                ] is not None
                else "N/A"
            ),
        },
        {
            "Metric": "Average Relevance (1-5)",
            "Result": (
                score_averages[
                    "human_relevance_1_5"
                ]
                if score_averages[
                    "human_relevance_1_5"
                ] is not None
                else "N/A"
            ),
        },
        {
            "Metric": "Average Hallucination Control (1-5)",
            "Result": (
                score_averages[
                    "human_hallucination_control_1_5"
                ]
                if score_averages[
                    "human_hallucination_control_1_5"
                ] is not None
                else "N/A"
            ),
        },
        {
            "Metric": "Average Scope Adherence (1-5)",
            "Result": (
                score_averages[
                    "human_scope_adherence_1_5"
                ]
                if score_averages[
                    "human_scope_adherence_1_5"
                ] is not None
                else "N/A"
            ),
        },
        {
            "Metric": "Average Tool Use (1-5)",
            "Result": (
                score_averages[
                    "human_tool_use_1_5"
                ]
                if score_averages[
                    "human_tool_use_1_5"
                ] is not None
                else "N/A"
            ),
        },
        {
            "Metric": "Average Conversation Context (1-5)",
            "Result": (
                score_averages[
                    "human_conversation_context_1_5"
                ]
                if score_averages[
                    "human_conversation_context_1_5"
                ] is not None
                else "N/A"
            ),
        },
    ]

    summary_df = pd.DataFrame(
        summary_rows
    )

    # ----------------------------------------------------------------
    # Failed cases
    #
    # A case is considered failed if:
    #   - human verdict = FAIL
    #   OR
    #   - deterministic_pass = FAIL
    # ----------------------------------------------------------------

    deterministic_failed = (
        df["deterministic_pass"]
        .astype(str)
        .str.upper()
        .eq("FAIL")
        if "deterministic_pass" in df.columns
        else pd.Series(
            False,
            index=df.index,
        )
    )

    human_failed = (
        df["human_verdict"]
        .map(normalize_status)
        .eq("FAIL")
        if "human_verdict" in df.columns
        else pd.Series(
            False,
            index=df.index,
        )
    )

    failed_mask = (
        deterministic_failed
        | human_failed
    )

    failed_df = df.loc[
        failed_mask
    ].copy()

    # ----------------------------------------------------------------
    # Test Results
    # ----------------------------------------------------------------

    test_results_df = df.copy()

    # ----------------------------------------------------------------
    # Output path
    # ----------------------------------------------------------------

    output_path = (
        evaluation_path.parent
        / "evaluation_report.xlsx"
    )

    # ----------------------------------------------------------------
    # Write workbook
    # ----------------------------------------------------------------

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        summary_df.to_excel(
            writer,
            index=False,
            sheet_name="Summary",
        )

        test_results_df.to_excel(
            writer,
            index=False,
            sheet_name="Test Results",
        )

        failed_df.to_excel(
            writer,
            index=False,
            sheet_name="Failed Cases",
        )

    print(
        "\nReport created successfully:"
    )
    print(
        f"  {output_path}"
    )

    print("\nSummary:")
    print(
        f"  Total tests              : {total_tests}"
    )
    print(
        f"  Deterministic pass rate  : "
        f"{deterministic_pass_rate}%"
    )
    print(
        f"  Human evaluated          : "
        f"{manual_evaluated_count}"
    )
    print(
        f"  Human passed             : "
        f"{manual_pass_count}"
    )
    print(
        f"  Human failed             : "
        f"{manual_fail_count}"
    )
    print(
        f"  Failed cases in report   : "
        f"{len(failed_df)}"
    )

    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "\nUsage:"
            "\n  python evaluation/generate_report.py "
            "<evaluation_results.xlsx>"
        )

        print(
            "\nExample:"
            "\n  python evaluation/generate_report.py "
            "evaluation/results/2026-10-11_02-57-31/"
            "evaluation_results.xlsx"
        )

        sys.exit(1)

    input_path = Path(
        sys.argv[1]
    ).resolve()

    generate_report(
        input_path
    )
```

The report intentionally remains simple: **Summary → Test Results → Failed Cases**, matching your proposed design. 

---

# 8. What the generated evaluation Excel looks like

After running the tests, you'll get:

```text
evaluation/results/
└── 2026-10-11_02-57-31/
    │
    ├── raw_results.json
    │
    └── evaluation_results.xlsx
```

The Excel will contain columns like:

```text
INPUT
────────────────────────────────────────
test_id
conversation_id
round
scope_type
memory_id
query
expected_answer
expected_memory_ids
expected_tools
max_tool_calls

AUTOMATIC RESULTS
────────────────────────────────────────
actual_answer
actual_memory_ids
actual_tools
tool_call_count
tool_limit_reached
blocked_tool_calls
actual_scope
actual_memory_id
retrieval_sources
thread_id
latency_ms
error

AUTOMATIC CHECKS
────────────────────────────────────────
answer_check
memory_check
tool_check
tool_budget_check
scope_isolation_check
deterministic_pass

MANUAL EVALUATION
────────────────────────────────────────
answer_correct
retrieval_correct
tool_correct
tool_budget_ok

human_correctness_1_5
human_groundedness_1_5
human_relevance_1_5
human_hallucination_control_1_5
human_scope_adherence_1_5
human_tool_use_1_5
human_conversation_context_1_5

human_verdict
human_comments
```

The manual columns are intentionally left blank.

---

# 9. Your manual evaluation process

Open:

```text
evaluation_results.xlsx
```

For every test case, inspect:

```text
query
expected_answer
actual_answer
expected_memory_ids
actual_memory_ids
expected_tools
actual_tools
```

Then fill:

### Binary evaluation

```text
answer_correct
retrieval_correct
tool_correct
tool_budget_ok
```

Use:

```text
PASS
FAIL
```

### 1–5 scoring

Use:

```text
1 = Very Poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent
```

For example:

| Metric                | What you evaluate                                 |
| --------------------- | ------------------------------------------------- |
| Correctness           | Is the final answer factually correct?            |
| Groundedness          | Is the answer supported by retrieved memory?      |
| Relevance             | Does it directly answer the query?                |
| Hallucination Control | Did it avoid inventing information?               |
| Scope Adherence       | Did it stay within selected memory scope?         |
| Tool Use              | Was the appropriate tool used?                    |
| Conversation Context  | Did it correctly use previous conversation turns? |

Then:

```text
human_verdict
```

should be:

```text
PASS
```

or

```text
FAIL
```

and use:

```text
human_comments
```

for your observations.

This gives you a human-authored evaluation baseline rather than prematurely adding an LLM judge.

---

# 10. Generate the final report

After manually filling the Excel, run:

```bash
python evaluation/generate_report.py evaluation/results/2026-10-11_02-57-31/evaluation_results.xlsx
```

Replace the timestamp with your actual folder.

It creates:

```text
evaluation/results/2026-10-11_02-57-31/
│
├── raw_results.json
├── evaluation_results.xlsx
└── evaluation_report.xlsx
```

---

# 11. Test the implementation step-by-step

## Step 1 — Check the project structure

From the project root:

```bash
cd memory-vault
```

You should have:

```text
memory-vault/
├── app.py
├── config/
├── database/
├── ingestion/
├── memory/
├── rag/
├── agent/
└── evaluation/
    ├── golden_questions.xlsx
    ├── run_tests.py
    └── generate_report.py
```

---

## Step 2 — Check your existing agent import

Before running the complete evaluation, verify that your existing graph can be imported:

```bash
python -c "from agent.graph import run_langgraph_agent; print('Agent import successful')"
```

Expected:

```text
Agent import successful
```

---

## Step 3 — Prepare only a few test cases first

Don't start with 50–100 cases.

Start with **4–5 cases**.

For example:

```text
T001
C001
1
memory
<real memory id>
What was the price mentioned in this memory?
<expected answer>
search_memories
4
```

Then:

```text
T002
C001
2
memory
<same memory id>
What else did I mention about it?
<expected answer>
search_memories
4
```

This specifically verifies that the second question gets the first conversation round as context.

---

## Step 4 — Test global retrieval

Add something like:

```text
T003
C002
1
all
What was the name of my hotel?
Hotel ABC
<hotel-memory-id>
search_memories
4
```

---

## Step 5 — Test a tool

For example:

```text
T004
C003
1
all
Calculate 25 + 75
100
calculate
4
```

---

## Step 6 — Test scope isolation

This is important for your Memory Vault architecture.

For example:

```text
test_id:
SEC-001

conversation_id:
SEC-C001

round:
1

scope_type:
memory

memory_id:
<HEADPHONE_MEMORY_ID>

query:
What was the name of my hotel?

expected_answer:
Information should not be found in this memory.

expected_memory_ids:
[blank]

expected_tools:
search_memories

max_tool_calls:
4
```

Your design specifically treats these security/regression scenarios as normal rows in the golden dataset rather than creating a separate security-testing script. 

---

# 12. Run the test runner

From the project root:

```bash
python evaluation/run_tests.py
```

You should see something similar to:

```text
======================================================================
Memory Vault Evaluation
======================================================================

Input file:
  .../evaluation/golden_questions.xlsx

Total test cases: 5

Output directory:
  .../evaluation/results/2026-10-11_02-57-31

[1/5] T001 | Conversation=C001 | Round=1
  Scope : memory
  Query : What was the headphone price?
  Memory: abc123
  Answer : ₹25,000
  Tools  : ['search_memories']
  Calls  : 1
  Memory : ['abc123']
  Time   : 1234.5 ms
  Result : PASS
```

At the end:

```text
======================================================================
Evaluation execution completed.
======================================================================

Raw results:
  .../raw_results.json

Evaluation Excel:
  .../evaluation_results.xlsx
```

---

# 13. Open the evaluation Excel

Open:

```text
evaluation/results/<timestamp>/evaluation_results.xlsx
```

Check that:

```text
actual_answer
actual_memory_ids
actual_tools
tool_call_count
latency_ms
```

are populated.

Then manually fill:

```text
answer_correct
retrieval_correct
tool_correct
tool_budget_ok

human_correctness_1_5
human_groundedness_1_5
human_relevance_1_5
human_hallucination_control_1_5
human_scope_adherence_1_5
human_tool_use_1_5
human_conversation_context_1_5

human_verdict
human_comments
```

---

# 14. Generate report

Once manual evaluation is complete:

```bash
python evaluation/generate_report.py evaluation/results/<timestamp>/evaluation_results.xlsx
```

You'll get:

```text
evaluation/results/<timestamp>/
│
├── raw_results.json
├── evaluation_results.xlsx
└── evaluation_report.xlsx
```

The report has:

```text
evaluation_report.xlsx

├── Summary
├── Test Results
└── Failed Cases
```

---

# 15. Final testing workflow

So your complete workflow is now simply:

```text
                YOU
                 │
                 ▼
     Manually upload memories
        through Memory Vault
                 │
                 ▼
       Manually create Excel
     golden_questions.xlsx
                 │
                 ▼
          run_tests.py
                 │
                 ▼
       ┌─────────┴─────────┐
       ▼                   ▼
raw_results.json   evaluation_results.xlsx
                           │
                           ▼
                  MANUAL EVALUATION
                           │
                           ▼
                 generate_report.py
                           │
                           ▼
                 evaluation_report.xlsx
```

This keeps the evaluation system separate from the core Memory Vault logic. It also preserves your existing backend-controlled scope instead of duplicating authorization logic in the evaluator, which is exactly the separation recommended in your design. 

**One important note:** the automatic `answer_check` is intentionally only a simple signal (for example, whether the expected answer text appears in the actual answer). It should **not** be treated as the final answer-quality metric. Your manual `human_*` evaluation is what determines the real quality of the agent.
