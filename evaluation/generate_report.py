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