Absolutely. I would keep your original product idea almost intact, but incorporate the architectural changes we discussed: **RAG over original extracted content, tool calling, agent routing, metadata filtering, grounded responses, and evaluation**.

Here is a cleaned-up version that can serve as your **official project specification/V1 requirements**.

# Project Idea: **Memory Vault — Personal Memory + RAG Assistant**

*Inspired by the Mind Space concept on OnePlus/OPPO phones*

> **Scope:** Memory Vault is a lightweight, solo-built AI application inspired by modern AI memory systems. It is not intended to replicate the complete functionality of OnePlus/OPPO Mind Space. The focus is on demonstrating practical AI engineering skills such as **multimodal processing, RAG, vector search, tool calling, AI agents, memory, grounding, and evaluation**, while keeping the application simple enough for a single developer to build.

---

# 1. Concept

**Memory Vault** is a personal AI memory assistant where users can upload **screenshots/photos, documents, and voice recordings**. Each uploaded item becomes a **memory** belonging to the user.

The application processes the uploaded content so that the actual information contained within images, documents, and audio can later be searched and queried.

The AI can:

* Understand uploaded content
* Generate a title and summary
* Store the content as a searchable memory
* Search memories using natural language
* Answer questions using information from the user's memories
* Identify the specific memories used to answer a question
* Refuse to answer when the required information is not present
* Optionally use external tools such as web search when the user explicitly asks for outside information

The primary goal is to create a **multimodal personal memory system with RAG and agentic capabilities**, rather than a complex consumer application.

---

# 2. Supported Memory Types

The application initially supports three types of memories:

### 1. Image Memories

Examples:

* Screenshots
* Photos
* Tickets
* Receipts
* Product screenshots
* Booking confirmations
* Notes photographed using a phone

The application uses a vision-capable AI model to understand the image and extract useful textual information.

For example:

```text
Flight Ticket Screenshot

Airline: IndiGo
Flight: 6E123
From: Hyderabad
To: Delhi
Date: 15 September
Time: 10:30 AM
```

The extracted information becomes searchable content for that memory.

---

### 2. Document Memories

Supported formats:

* PDF
* DOCX
* TXT

The application extracts text from the document and processes it for semantic search.

```text
Document
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Database
```

---

### 3. Voice Memories

The user can upload a voice recording.

```text
Audio
  ↓
Speech-to-Text
  ↓
Transcript
  ↓
Chunking
  ↓
Embeddings
  ↓
Vector Database
```

The transcript becomes the searchable content of the memory.

---

# 3. Memory Processing Pipeline

Regardless of the input type, all memories eventually follow a common pipeline.

```text
Image ──────┐
PDF ────────┤
DOCX ───────┤
TXT ────────┤
Audio ──────┘
      ↓
Content Extraction
      ↓
Normalized Text
      ↓
Chunking
      ↓
Embeddings
      ↓
Vector Database
```

The original file is retained separately so that the user can view or play the original memory.

This separation is important because the **original extracted content**, rather than only the generated summary, is used for RAG.

---

# 4. Memory Metadata

Every memory contains basic metadata.

For example:

```text
Memory ID
User ID
Memory Type
File Path
Title
Summary
Created At
```

The vector-store chunks additionally contain metadata such as:

```text
memory_id
user_id
memory_type
category
created_at
source
```

This metadata allows the system to perform searches such as:

* Search all memories
* Search only one memory
* Search only image memories
* Search only documents
* Search memories belonging to the current user

It also ensures that **one user's memories cannot be retrieved while another user is asking a question.**

---

# 5. Dashboard

The dashboard displays all memories belonging to the current user.

Memories can be grouped into:

* **All**
* **Photo Memories**
* **Document Memories**
* **Voice Memories**

Each memory displays:

* Type icon
* Title
* Upload date/time
* Optional short summary

If an AI-generated title has not yet been created, the application displays the title provided by the user during upload.

The dashboard also contains a central:

> **Ask Question**

button.

---

# 6. Memory Detail Page

Clicking a memory opens its detail page.

The page displays:

* Original image/document/audio
* Memory title
* Summary, if generated
* Upload date
* Memory type

It also provides:

### Generate Summary

When the user clicks **Generate Summary**, the AI processes the memory and generates:

```text
Title
Summary
```

Both are stored in the database so they do not need to be regenerated every time the user opens the memory.

The summary is primarily intended for **browsing and understanding the memory**.

**Important:** The summary is not the primary RAG source.

The original extracted content remains the source used for retrieval.

---

# 7. Content Safety Check

Before generating a summary, the application performs a basic content-safety check.

If the uploaded memory contains prohibited or unsafe content, the application does not generate a summary and instead displays a message such as:

> **"Summary can't be generated for this content."**

The same general safety principles should apply to the rest of the AI functionality.

The system should also prevent:

* Cross-user memory leakage
* Fabricated memory references
* Unsupported claims presented as memory information

---

# 8. Ask Question — Two Scopes

The application supports two different question scopes.

## A. Dashboard-Level Question

When the user asks a question from the dashboard, the system searches across **all of the user's memories**.

For example:

> "What was the hotel I booked for my Delhi trip?"

The system searches the user's memories and retrieves relevant information.

The answer should also identify the source memory.

Example:

> **You booked Hotel ABC for your Delhi trip.**
> Source: *Delhi Hotel Booking Screenshot*

The source should link back to the corresponding memory detail page.

---

## B. Memory-Level Question

When the user asks a question from a specific memory's detail page, the search is restricted to that memory.

For example:

> "What is the total amount mentioned in this receipt?"

The retriever should only search the content associated with that memory.

This prevents unrelated memories from influencing the answer.

---

# 9. RAG Architecture

The core question-answering system uses **Retrieval-Augmented Generation (RAG)**.

The basic flow is:

```text
User Question
      ↓
Query Processing
      ↓
Vector Search
      ↓
Retrieve Relevant Chunks
      ↓
Metadata Filtering
      ↓
Relevance / Grounding Check
      ↓
LLM
      ↓
Answer + Source Memory
```

The system retrieves information from the user's **original extracted memory content**, not merely from AI-generated summaries.

---

# 10. Grounded Answering

One of the most important requirements is:

> **The AI must not fabricate information that isn't present in the user's memories.**

For example, suppose a memory contains:

```text
iPhone 16
Price: ₹70,000
```

User asks:

> "What price was mentioned for the iPhone 16?"

The system should answer:

> "The price mentioned in your memory is ₹70,000."

But suppose the memory only says:

```text
iPhone 16
```

and doesn't contain a price.

The system should say:

> **"This information wasn't found in your memory."**

It should **not** search the internet and return the current/general price unless the user explicitly asks for outside information.

---

# 11. Retrieval Confidence Check

The application should not rely only on prompting the LLM with:

> "Don't hallucinate."

Instead, the retrieval pipeline performs a relevance check.

Conceptually:

```text
Question
   ↓
Retriever
   ↓
Top K Chunks
   ↓
Similarity Scores
   ↓
Are relevant chunks available?
       │
    ┌──┴──┐
   YES    NO
    │      │
    ↓      ↓
   LLM   "Not found
          in memory"
```

If retrieved information does not meet the required relevance threshold, the system should not generate a memory-grounded answer.

This provides a stronger defense against hallucination.

---

# 12. AI Agent and Tool Calling

Memory Vault will use a **LangGraph-based AI agent** to decide when different tools are required.

Rather than having every question follow exactly the same pipeline, the agent determines which tool or tools are appropriate.

The initial tool set will remain small:

### Tool 1 — `search_memories`

Search the user's memories using semantic search.

```text
search_memories(query)
```

---

### Tool 2 — `get_memory`

Retrieve a specific memory and its associated information.

```text
get_memory(memory_id)
```

---

### Tool 3 — `search_web`

Search external information when the user explicitly asks for current or outside knowledge.

```text
search_web(query)
```

---

### Tool 4 — `calculate`

Perform numerical calculations when required.

```text
calculate(expression)
```

The project does not need a large number of tools. The goal is to demonstrate **meaningful tool usage**, not tool quantity.

---

# 13. Agent Routing

The agent can determine what type of question the user is asking.

For example:

### Memory-only question

> "What was the price in the screenshot?"

```text
User
 ↓
Agent
 ↓
search_memories()
 ↓
Retrieved Memory
 ↓
Answer
```

### Memory + external information

> "What was the price in my screenshot and what is the current price?"

```text
User
 ↓
Agent
 ↓
search_memories()
 ↓
search_web()
 ↓
Compare Information
 ↓
Answer
```

### Calculation

> "What is the total cost of the three products I saved?"

```text
User
 ↓
Agent
 ↓
search_memories()
 ↓
calculate()
 ↓
Answer
```

This gives LangGraph and tool calling a genuine purpose in the architecture.

---

# 14. Memory Isolation

Every memory belongs to a specific user.

Therefore, all retrieval operations must respect:

```text
user_id
```

For example:

```text
User A
 ↓
search_memories()
 ↓
Only User A's memories
```

User B should never be able to retrieve User A's information.

Memory-level queries additionally filter by:

```text
memory_id
```

when the question is asked from a particular memory.

---

# 15. Structured + Semantic Memory

Memory Vault will use two complementary storage approaches.

### SQL Database

Used for application and structured information:

```text
Users
Memories
Titles
Summaries
Metadata
```

### Vector Database

Used for semantic retrieval:

```text
Memory chunks
Embeddings
Metadata
```

Conceptually:

```text
                 MEMORY
                    │
           ┌────────┴────────┐
           ↓                 ↓
      SQL Database       Vector DB
           │                 │
      Metadata         Semantic Search
           │                 │
           └────────┬────────┘
                    ↓
                  Agent
```

This avoids trying to use a vector database for everything.

---

# 16. Optional External Knowledge

Memory Vault should primarily answer from the user's memories.

However, the system can optionally provide external knowledge through the web-search tool.

The distinction must be explicit.

For example:

> **From your memory:**
> The hotel price you saved was ₹5,000/night.

> **From external information:**
> The current listed price is ₹6,200/night.

This prevents external information from being incorrectly presented as something the user previously saved.

---

# 17. LangGraph Persistence

The AI assistant can use LangGraph persistence to maintain conversation state.

This provides **short-term conversational memory**.

For example:

```text
User:
I'm planning a Delhi trip.

Assistant:
Okay.

User:
What hotel did I save?

Assistant:
You saved Hotel ABC.

User:
How much was it?

Assistant:
The saved price was ₹5,000.
```

The assistant can maintain the context of the ongoing conversation through a LangGraph thread/checkpointer.

This is separate from the user's **long-term stored memories**.

---

# 18. Human-in-the-Loop — Optional

A lightweight Human-in-the-Loop feature can be added later for information extraction.

For example:

```text
AI extracted:

Flight: 6E123
From: Hyderabad
To: Delhi
Date: 15 September

[Save] [Edit] [Reject]
```

The user can approve or modify the extracted information before it is stored.

This feature is optional for V1 and should only be implemented after the core RAG and agent functionality is complete.

---

# 19. Evaluation and Observability

The project should include basic evaluation rather than relying entirely on manual testing.

A small evaluation dataset can contain questions such as:

```text
Question                         Expected Result
--------------------------------------------------
What is the flight number?       6E123
What hotel did I book?           Hotel ABC
What was the price?              ₹8,500
What date is the flight?         15 September
```

The system can be evaluated on:

### Retrieval relevance

Did the retriever find the correct memory/chunk?

### Answer correctness

Did the generated answer match the expected information?

### Groundedness

Was the answer actually supported by the retrieved memory?

### Refusal accuracy

When information does not exist, did the system correctly say:

> "This information wasn't found in your memory."

LangSmith can be used for:

* Tracing
* Debugging
* Retrieval inspection
* LLM evaluation
* Agent/tool-call monitoring

---

# 20. Overall Architecture

The final V1 architecture would look like this:

```text
                         MEMORY VAULT
                              │
                              ▼
                         Simple UI
                         Streamlit
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
          Image            Document           Audio
            │                 │                 │
            ▼                 ▼                 ▼
        Vision LLM       Text Extraction       STT
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
                       Extracted Content
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
              Title/Summary         Chunking
                    │                   │
                    ▼                   ▼
               SQL Database        Embeddings
                                        │
                                        ▼
                                   Vector DB
                                        │
                                        ▼
                                  LangGraph Agent
                                        │
                         ┌──────────────┼──────────────┐
                         │              │              │
                         ▼              ▼              ▼
                  Search Memory    Get Memory      Web Search
                         │              │              │
                         └──────────────┼──────────────┘
                                        │
                                        ▼
                                Grounding Check
                                        │
                                        ▼
                                   LLM Response
                                        │
                                  ┌─────┴─────┐
                                  ▼           ▼
                               Answer      Sources
```

With LangSmith monitoring the AI pipeline:

```text
                    LangSmith
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       Tracing      Evaluation     Debugging
```

---

# 21. V1 Feature Priorities

To keep the project manageable as a solo developer:

### Must Have

* Image upload
* PDF/DOCX/TXT upload
* Voice upload
* Content extraction
* AI-generated title + summary
* Vector database
* RAG
* Metadata filtering
* Dashboard-level search
* Memory-level search
* Source references
* Grounded responses
* "Not found in memory" behavior
* LangGraph agent
* Tool calling
* SQL + Vector DB
* Basic LangSmith tracing/evaluation

### Nice to Have

* Web search
* Calculator tool
* Human-in-the-loop
* Persistent conversation threads

### Don't Build

* Native Android/iOS app
* Complex frontend
* Knowledge graph
* Complex authentication system
* Multiple-agent architecture
* Custom AI model
* Fine-tuning
* Complicated cloud infrastructure

---

# 22. Core AI Technologies Demonstrated

The project will allow you to demonstrate:

| Technology             | Where it is used               |
| ---------------------- | ------------------------------ |
| **LLM**                | Summarization, answering       |
| **Vision LLM**         | Image/screenshot understanding |
| **Speech-to-Text**     | Voice memories                 |
| **Embeddings**         | Semantic representation        |
| **Vector DB**          | Memory retrieval               |
| **RAG**                | Question answering             |
| **Metadata filtering** | Memory/user isolation          |
| **LangChain**          | LLM/RAG/tool components        |
| **LangGraph**          | Agent workflow                 |
| **Tool Calling**       | Memory, web, calculation       |
| **Short-term memory**  | Conversation state             |
| **Long-term memory**   | Stored user memories           |
| **Persistence**        | LangGraph checkpointer         |
| **HITL**               | Optional extraction approval   |
| **LangSmith**          | Tracing & evaluation           |
| **SQL**                | Structured memory metadata     |

---

## The most important design decision

If I had to reduce the entire project to one architecture principle, it would be:

> **Memory Vault should be a RAG-first system with an agent on top of it — not an agent-first system with RAG somewhere inside it.**

Your **memory retrieval and grounding** are the core product. The agent exists to decide **when and how to use that memory and other tools**.

That keeps the project technically meaningful **without turning it into an unnecessarily complicated multi-agent system**.

This scope is also very well aligned with the AI topics you've been learning, because almost every major concept you've studied can have a real purpose in this one project.
