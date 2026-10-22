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