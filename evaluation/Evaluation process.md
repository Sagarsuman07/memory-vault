Yes. The easiest way to understand this is to divide the 44 columns into **4 groups**:

1. **You manually prepare before running tests**
2. **`run_tests.py` automatically fills**
3. **You manually evaluate after the test runs**
4. **`generate_report.py` uses the completed evaluation**

Your original input Excel should contain only the first group. The remaining columns are generated later by the runner. This follows the simplified evaluation design you selected. 

---

# 1. Columns you manually fill BEFORE running tests

These are the **10 columns** you need to prepare in `golden_questions.xlsx`.

| Column                | Required?      | What you should fill                    |
| --------------------- | -------------- | --------------------------------------- |
| `test_id`             | ✅ Yes          | Unique ID for each test                 |
| `conversation_id`     | ✅ Yes          | ID for grouping multi-turn conversation |
| `round`               | ✅ Yes          | Conversation round number               |
| `scope_type`          | ✅ Yes          | `all` or `memory`                       |
| `memory_id`           | ⚠️ Conditional | Memory ID when scope is `memory`        |
| `query`               | ✅ Yes          | The actual user question                |
| `expected_answer`     | ✅ Yes          | What you expect the agent to answer     |
| `expected_memory_ids` | ⚠️ Conditional | Expected memory ID(s) for `all` scope   |
| `expected_tools`      | ⚠️ Usually yes | Tool(s) you expect the agent to use     |
| `max_tool_calls`      | ✅ Yes          | Maximum allowed tool calls              |

Let's go through each one.

---

## 1. `test_id`

### What is it?

A unique identifier for the test case.

### What to fill?

Use something simple:

```text
T001
T002
T003
T004
```

For security-related tests, you can use:

```text
SEC-001
SEC-002
```

### Example

```text
T001
```

### Important

Every test should have a different `test_id`.

---

# 2. `conversation_id`

### What is it?

This tells the evaluator which questions belong to the **same conversation**.

For a single-turn question:

```text
C001
```

For another independent conversation:

```text
C002
```

### Example

Suppose you want to test:

> What was the price of my headphones?

Then:

> When did I buy them?

These should be:

| test_id | conversation_id | round |
| ------- | --------------- | ----: |
| T001    | C001            |     1 |
| T002    | C001            |     2 |

Because both belong to the same conversation.

The runner will use the same LangGraph thread for these rounds. Your implementation design specifically calls for this behavior. 

### If questions are independent

Use different conversation IDs:

| test_id | conversation_id | round |
| ------- | --------------- | ----: |
| T001    | C001            |     1 |
| T002    | C002            |     1 |
| T003    | C003            |     1 |

---

# 3. `round`

### What is it?

The sequence number inside a conversation.

For:

```text
C001
```

you might have:

```text
Round 1
Round 2
Round 3
```

So:

| test_id | conversation_id | round |
| ------- | --------------- | ----: |
| T001    | C001            |     1 |
| T002    | C001            |     2 |
| T003    | C001            |     3 |

For independent questions, always use:

```text
1
```

### What to fill?

Just enter:

```text
1
2
3
...
```

---

# 4. `scope_type`

### What is it?

It defines **which memories the agent is allowed to search**.

There are only two values:

```text
all
```

or

```text
memory
```

### `all`

Means:

> Search across all memories belonging to the user.

Example:

```text
scope_type = all
```

### `memory`

Means:

> Search only inside one specific memory.

Example:

```text
scope_type = memory
memory_id = abc123
```

This is especially important for testing your scope-isolation/security behavior.

---

# 5. `memory_id`

### Required?

**Only when `scope_type = memory`.**

If:

```text
scope_type = memory
```

then:

```text
memory_id = <actual memory ID>
```

For example:

```text
memory_id = 26da0508-b550-484a-8f72-e20481e1ce87
```

If:

```text
scope_type = all
```

leave it **blank**.

### Example

```text
scope_type | memory_id
-----------|------------------------------------
memory     | 26da0508-b550-484a-8f72-e20481e1ce87
```

or:

```text
scope_type | memory_id
-----------|-----------
all        |
```

---

# 6. `query`

### What is it?

This is the **actual question you want to send to Memory Vault**.

This should be exactly how a user would ask the question.

### Example

```text
What was the price of the headphones?
```

Don't put instructions such as:

```text
Use search_memories and answer correctly.
```

Instead, write the natural user query:

```text
What was the price of the headphones?
```

---

# 7. `expected_answer`

### What is it?

This is what **you expect Memory Vault to answer**.

It is your golden/reference answer.

### Example

If the memory contains:

> Sony WH-1000XM5 purchased for ₹25,000.

Then:

```text
expected_answer = ₹25,000
```

You don't necessarily need to write the entire perfect response.

For example:

```text
₹25,000
```

is sufficient.

The current runner uses this primarily as a **simple automatic comparison**. It is not intended to replace your human evaluation.

### For a "not found" test

Suppose the memory doesn't contain the requested information.

You can write:

```text
Information wasn't found in this memory.
```

or your expected grounded response.

This is especially useful for testing that Memory Vault doesn't hallucinate.

---

# 8. `expected_memory_ids`

### Required?

Primarily when:

```text
scope_type = all
```

and the answer is expected to come from one or more particular memories.

### Example

Suppose:

```text
query = What was my hotel name?
```

and you know the answer exists in:

```text
memory_id = HOTEL-123
```

Then:

```text
expected_memory_ids = HOTEL-123
```

If two memories should contribute:

```text
HOTEL-123,TRIP-456
```

### For `scope_type = memory`

You can normally leave this blank because `memory_id` already defines the allowed memory.

---

# 9. `expected_tools`

### What is it?

Which tool(s) you **expect the agent to use**.

Your current Memory Vault agent has tools such as:

```text
search_memories
calculate
search_web
```

### Example 1 — Memory question

```text
expected_tools = search_memories
```

### Example 2 — Calculation

```text
expected_tools = calculate
```

For:

```text
Calculate 25 + 75
```

### Example 3 — Web question

```text
expected_tools = search_web
```

### Multiple tools

If you expect multiple tools:

```text
search_memories,calculate
```

### Important

Don't put:

```text
LLM
agent
LangGraph
```

These aren't the tools being evaluated in this column.

---

# 10. `max_tool_calls`

### What is it?

The maximum number of tool calls you allow for that particular test.

Your current application configuration uses:

```text
MAX_TOOL_CALLS = 4
```

So for most tests you can simply use:

```text
4
```

Example:

```text
max_tool_calls = 4
```

For a simple calculation, you could still use:

```text
4
```

You don't need to make it `1` unless you specifically want to test a stricter budget.

---

# 2. Columns automatically filled by `run_tests.py`

**You do NOT fill these manually.**

The runner fills them after executing the test.

---

## 11. `actual_answer`

The actual response produced by Memory Vault.

Example:

```text
₹25,000
```

You compare this with:

```text
expected_answer
```

---

## 12. `actual_memory_ids`

Which memories were actually retrieved/used as sources.

Example:

```text
abc123
```

or:

```text
abc123,def456
```

You compare this with:

```text
expected_memory_ids
```

---

## 13. `actual_tools`

Which tools the agent actually called.

Example:

```text
search_memories
```

or:

```text
calculate
```

---

## 14. `tool_call_count`

How many tool calls actually happened.

Example:

```text
1
```

or:

```text
3
```

This is automatically captured.

---

## 15. `tool_limit_reached`

Whether the agent reached its configured tool limit.

Example:

```text
False
```

or:

```text
True
```

You don't enter this.

---

## 16. `blocked_tool_calls`

If the agent attempted to call a tool after reaching the limit, the blocked calls are recorded here.

Usually:

```text
[]
```

means nothing was blocked.

---

## 17. `actual_scope`

The scope actually returned by the agent.

For example:

```text
all
```

or the corresponding scope representation from your graph.

This lets you verify that the execution used the expected scope.

---

## 18. `actual_memory_id`

The actual memory ID passed to the agent.

For:

```text
scope_type = memory
```

this should contain the selected memory ID.

For:

```text
scope_type = all
```

it should normally be empty.

---

## 19. `retrieval_sources`

Detailed retrieval-source information returned by the agent.

For example, it can contain information like:

```text
memory_id
memory_type
title
distance
```

This is useful when investigating why a particular answer was generated.

---

## 20. `thread_id`

The LangGraph conversation thread used by the test.

For example:

```text
demo_user:550e8400-e29b-41d4-a716-446655440000
```

You normally don't need to evaluate this manually.

It is mainly useful for debugging.

---

## 21. `latency_ms`

How long the query took.

Example:

```text
1250.43
```

meaning approximately:

```text
1.25 seconds
```

This is useful for performance analysis.

---

## 22. `error`

If the test execution failed, the error is recorded here.

Normally:

```text
```

(blank)

If something fails:

```text
ValueError: ...
```

---

# 3. Automatically calculated check columns

These are also **not manually filled**.

---

## 23. `answer_check`

A simple automatic comparison between:

```text
expected_answer
```

and:

```text
actual_answer
```

Possible values:

```text
PASS
FAIL
NOT_APPLICABLE
```

### Important

Don't treat this as the final correctness result.

For example:

```text
Expected:
₹25,000

Actual:
The headphones cost approximately ₹25,000.
```

This could pass.

But semantic correctness can be more complicated.

That's why you still have:

```text
answer_correct
human_correctness_1_5
```

---

# 24. `memory_check`

Compares:

```text
expected_memory_ids
```

with:

```text
actual_memory_ids
```

Example:

```text
expected:
MEM001

actual:
MEM001
```

→

```text
PASS
```

If the expected memory isn't retrieved:

```text
FAIL
```

---

# 25. `tool_check`

Compares:

```text
expected_tools
```

with:

```text
actual_tools
```

Example:

```text
expected_tools = search_memories
actual_tools   = search_memories
```

→

```text
PASS
```

---

# 26. `tool_budget_check`

Checks:

```text
actual tool calls <= max_tool_calls
```

Example:

```text
max_tool_calls = 4
tool_call_count = 2
```

→

```text
PASS
```

But:

```text
max_tool_calls = 4
tool_call_count = 5
```

→

```text
FAIL
```

---

# 27. `scope_isolation_check`

This is particularly important for your Memory Vault.

For example:

```text
scope_type = memory
memory_id = MEMORY_A
```

If the agent retrieves:

```text
MEMORY_A
```

→

```text
PASS
```

If it somehow retrieves:

```text
MEMORY_B
```

→

```text
FAIL
```

This gives you an automatic signal for scope leakage.

---

# 28. `deterministic_pass`

This is the overall **automatic** result.

It combines the applicable automatic checks.

For example:

```text
answer_check          PASS
memory_check          PASS
tool_check            PASS
tool_budget_check     PASS
scope_isolation_check PASS
```

Therefore:

```text
deterministic_pass = PASS
```

But remember:

> **This is not your final evaluation.**

Your human evaluation is still important.

---

# 4. Columns YOU manually fill AFTER running tests

These are the most important columns for your actual evaluation.

---

# 29. `answer_correct`

Ask:

> Is the actual answer correct?

Fill:

```text
PASS
```

or:

```text
FAIL
```

### Example

Expected:

```text
₹25,000
```

Actual:

```text
The headphones cost ₹25,000.
```

→

```text
PASS
```

If actual says:

```text
₹30,000
```

→

```text
FAIL
```

---

# 30. `retrieval_correct`

Ask:

> Did the agent retrieve the correct memory/memories needed to answer the question?

Example:

```text
expected_memory_ids:
MEM001

actual_memory_ids:
MEM001
```

→

```text
PASS
```

If it retrieves an unrelated memory:

```text
MEM005
```

→

```text
FAIL
```

This is different from answer correctness.

An answer can accidentally be correct even when the wrong memory was retrieved.

---

# 31. `tool_correct`

Ask:

> Did the agent use the appropriate tool?

Example:

```text
Query:
What was my headphone price?

Expected:
search_memories

Actual:
search_memories
```

→

```text
PASS
```

For:

```text
Calculate 25 + 75
```

expected:

```text
calculate
```

actual:

```text
calculate
```

→

```text
PASS
```

---

# 32. `tool_budget_ok`

Ask:

> Did the agent stay within the allowed tool-call limit?

Example:

```text
max_tool_calls = 4
actual = 2
```

→

```text
PASS
```

If actual = 5:

```text
FAIL
```

---

# 33. `human_correctness_1_5`

Rate the actual answer from **1 to 5**.

I recommend:

| Score | Meaning              |
| ----: | -------------------- |
|     1 | Completely incorrect |
|     2 | Mostly incorrect     |
|     3 | Partially correct    |
|     4 | Mostly correct       |
|     5 | Fully correct        |

Example:

```text
5
```

if the answer is completely correct.

---

# 34. `human_groundedness_1_5`

Ask:

> Is the answer actually supported by the retrieved memory?

| Score | Meaning                |
| ----: | ---------------------- |
|     1 | Completely unsupported |
|     2 | Mostly unsupported     |
|     3 | Partially supported    |
|     4 | Mostly supported       |
|     5 | Fully supported        |

This is especially important for your RAG system.

---

# 35. `human_relevance_1_5`

Ask:

> Does the answer directly answer the user's question?

Example:

User:

> What was the price?

Bad:

> The headphones were purchased in September and were black.

Good:

> They cost ₹25,000.

So the second would get a higher relevance score.

---

# 36. `human_hallucination_control_1_5`

Ask:

> Did the agent avoid inventing information?

| Score | Meaning                               |
| ----: | ------------------------------------- |
|     1 | Severe hallucination                  |
|     2 | Significant hallucination             |
|     3 | Some unsupported information          |
|     4 | Minor/no meaningful hallucination     |
|     5 | Completely grounded; no hallucination |

This is **very important for Memory Vault**, because one of your core requirements is:

> If the answer isn't in memory, don't guess.

---

# 37. `human_scope_adherence_1_5`

Ask:

> Did the agent respect the selected memory scope?

For example:

```text
Selected memory:
Headphones

Question:
What was my hotel name?
```

If the agent answers using a hotel memory outside the selected memory, that's a serious scope violation.

Suggested scoring:

| Score | Meaning                   |
| ----: | ------------------------- |
|     1 | Completely violated scope |
|     2 | Major scope violation     |
|     3 | Some concern              |
|     4 | Mostly respected          |
|     5 | Fully respected           |

---

# 38. `human_tool_use_1_5`

Ask:

> Did the agent choose and use tools appropriately?

For example:

```text
Calculate 25 + 75
```

Using:

```text
calculate
```

would deserve a high score.

Calling:

```text
search_web
```

for a simple calculation would be poor tool selection.

---

# 39. `human_conversation_context_1_5`

This is important for multi-round conversations.

Suppose:

### Round 1

> What was my headphone price?

Agent:

> ₹25,000.

### Round 2

> When did I buy them?

If the agent understands that **"them" = headphones**, that's good conversation-context handling.

For single-turn questions, you can leave this blank or mark it as not applicable according to your evaluation convention.

---

# 40. `human_verdict`

This is your **final manual decision**.

Use:

```text
PASS
```

or:

```text
FAIL
```

Think of this as:

> After considering everything, did this test case behave correctly?

---

# 41. `human_comments`

Use this for your observations.

Examples:

```text
Correct answer and correct memory retrieved.
```

or:

```text
Answer was correct, but the agent used search_web unnecessarily.
```

or:

```text
Agent leaked information from another memory.
```

or:

```text
Correct answer, but conversation context from round 1 was not used.
```

This becomes very useful when you analyze failed cases later.

---

# 42. The most important distinction

Your workflow should be:

### BEFORE execution

You fill:

```text
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
```

### Runner automatically fills

```text
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

answer_check
memory_check
tool_check
tool_budget_check
scope_isolation_check
deterministic_pass
```

### AFTER execution

You manually fill:

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

# 43. Your `golden_questions.xlsx` should therefore be much simpler

**Do not manually create all 44 columns.**

For the initial input file, create only:

| test_id | conversation_id | round | scope_type | memory_id | query                         | expected_answer | expected_memory_ids | expected_tools  | max_tool_calls |
| ------- | --------------- | ----: | ---------- | --------- | ----------------------------- | --------------- | ------------------- | --------------- | -------------: |
| T001    | C001            |     1 | memory     | MEM001    | What was the headphone price? | ₹25,000         |                     | search_memories |              4 |
| T002    | C001            |     2 | memory     | MEM001    | When did I buy them?          | September 2026  |                     | search_memories |              4 |
| T003    | C002            |     1 | all        |           | What was my hotel name?       | Hotel ABC       | HOTEL001            | search_memories |              4 |
| T004    | C003            |     1 | all        |           | Calculate 25 + 75             | 100             |                     | calculate       |              4 |

Then run:

```powershell
python evaluation/run_tests.py
```

The program creates the **full evaluation Excel** with the remaining columns.

After you manually evaluate that file, run:

```powershell
python evaluation/generate_report.py evaluation/results/<timestamp>/evaluation_results.xlsx
```

This keeps your workflow exactly as intended: **you only author the golden test cases; the system captures execution data; then you manually judge quality and generate the final report.** 
