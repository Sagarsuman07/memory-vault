import re

from langsmith import Client
from langsmith import traceable

from memory.memory_service import (
    answer_question,
)

from rag.chunker import split_text
from rag.retriever import retrieve
from rag.vector_store import index_memory


DATASET_NAME = (
    "Memory Vault RAG Evaluation V1"
)


EVAL_USER_ID = (
    "memory_vault_evaluation_user"
)


# ============================================================
# Normalize Text
# ============================================================

def normalize_text(
    text: str,
) -> str:

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    return " ".join(
        text.split()
    )


# ============================================================
# Index Evaluation Memory
# ============================================================

def index_evaluation_memory(
    memory_id: str,
    memory_text: str,
):

    # --------------------------------------------------------
    # Use the same chunker as the real application
    # --------------------------------------------------------

    chunks = split_text(
        memory_text
    )

    if not chunks:

        raise ValueError(
            f"No chunks generated for {memory_id}."
        )

    # --------------------------------------------------------
    # Use the same Chroma indexing pipeline
    # --------------------------------------------------------

    index_memory(
        memory_id=memory_id,
        user_id=EVAL_USER_ID,
        memory_type="document",
        chunks=chunks,
    )


# ============================================================
# Evaluation Target
# ============================================================

@traceable(
    name="Memory Vault Evaluation Target"
)
def target(
    inputs: dict,
):

    memory_id = inputs[
        "memory_id"
    ]

    memory_text = inputs[
        "memory_text"
    ]

    question = inputs[
        "question"
    ]

    # ========================================================
    # Prepare evaluation memory
    # ========================================================

    index_evaluation_memory(
        memory_id=memory_id,
        memory_text=memory_text,
    )

    # ========================================================
    # Run the REAL Memory Vault QA pipeline
    # ========================================================

    result = answer_question(
        question=question,
        user_id=EVAL_USER_ID,
        memory_id=memory_id,
    )

    # ========================================================
    # Retrieve raw chunks for evaluation
    # ========================================================

    raw_results = retrieve(
        question=question,
        user_id=EVAL_USER_ID,
        top_k=3,
        memory_id=memory_id,
    )

    retrieved_chunks = []

    for document, distance in raw_results:

        retrieved_chunks.append(
            {
                "memory_id":
                    document.metadata[
                        "memory_id"
                    ],

                "memory_type":
                    document.metadata[
                        "memory_type"
                    ],

                "chunk_index":
                    document.metadata[
                        "chunk_index"
                    ],

                "distance":
                    float(distance),

                "content":
                    document.page_content,
            }
        )

    # ========================================================
    # Return structured evaluation output
    # ========================================================

    return {

        "answer":
            result.get(
                "answer",
                "",
            ),

        "grounded":
            result.get(
                "grounded",
                False,
            ),

        "sources":
            result.get(
                "sources",
                [],
            ),

        "retrieved_chunks":
            retrieved_chunks,
    }


# ============================================================
# Retrieval Evaluator
# ============================================================

def retrieval_correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
):

    expected_found = reference_outputs[
        "expected_found"
    ]

    expected_evidence = normalize_text(
        reference_outputs.get(
            "expected_evidence",
            "",
        )
    )

    forbidden_evidence = normalize_text(
        reference_outputs.get(
            "forbidden_evidence",
            "",
        )
    )

    # --------------------------------------------------------
    # Combine retrieved chunks into one text
    # --------------------------------------------------------

    retrieved_text = normalize_text(
        " ".join(
            chunk["content"]
            for chunk in outputs.get(
                "retrieved_chunks",
                [],
            )
        )
    )

    # ========================================================
    # Answerable Question
    # ========================================================

    if expected_found:

        score = bool(
            expected_evidence
            and expected_evidence
            in retrieved_text
        )

    # ========================================================
    # Not Found Question
    #
    # Forbidden evidence should not be retrieved.
    # ========================================================

    else:

        if not forbidden_evidence:

            score = not outputs.get(
                "grounded",
                False,
            )

        else:

            score = (
                forbidden_evidence
                not in retrieved_text
            )

    return {

        "key":
            "retrieval_correctness",

        "score":
            1.0 if score else 0.0,
    }


# ============================================================
# Answer Evaluator
# ============================================================

def answer_correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
):

    expected_found = reference_outputs[
        "expected_found"
    ]

    expected_answer = normalize_text(
        reference_outputs.get(
            "expected_answer",
            "",
        )
    )

    actual_answer = normalize_text(
        outputs.get(
            "answer",
            "",
        )
    )

    # ========================================================
    # Answerable Question
    # ========================================================

    if expected_found:

        score = bool(
            expected_answer
            and expected_answer
            in actual_answer
        )

    # ========================================================
    # Not Found Question
    # ========================================================

    else:

        refusal_phrases = [

            "information wasn't found",

            "information was not found",

            "not found in your memory",

            "not found",

            "i couldn't find",

            "cannot find",

            "don't have that information",
        ]

        score = any(
            phrase in actual_answer
            for phrase in refusal_phrases
        )

    return {

        "key":
            "answer_correctness",

        "score":
            1.0 if score else 0.0,
    }


# ============================================================
# Grounding Evaluator
# ============================================================

def grounding_correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
):

    expected_found = reference_outputs[
        "expected_found"
    ]

    expected_evidence = normalize_text(
        reference_outputs.get(
            "expected_evidence",
            "",
        )
    )

    retrieved_text = normalize_text(
        " ".join(
            chunk["content"]
            for chunk in outputs.get(
                "retrieved_chunks",
                [],
            )
        )
    )

    actual_answer = normalize_text(
        outputs.get(
            "answer",
            "",
        )
    )

    # ========================================================
    # Not Found Case
    #
    # A correct refusal should not be grounded.
    # ========================================================

    if not expected_found:

        score = (
            outputs.get(
                "grounded",
                False,
            )
            is False
        )

    # ========================================================
    # Answerable Case
    #
    # Expected evidence must exist in:
    #
    # 1. Retrieved context
    # 2. Final answer
    # ========================================================

    else:

        score = (
            bool(expected_evidence)

            and expected_evidence
            in retrieved_text

            and expected_evidence
            in actual_answer
        )

    return {

        "key":
            "grounding_correctness",

        "score":
            1.0 if score else 0.0,
    }


# ============================================================
# Refusal Evaluator
# ============================================================

def refusal_correctness(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
):

    expected_found = reference_outputs[
        "expected_found"
    ]

    actual_answer = normalize_text(
        outputs.get(
            "answer",
            "",
        )
    )

    refusal_phrases = [

        "information wasn't found",

        "information was not found",

        "not found in your memory",

        "not found",

        "i couldn't find",

        "cannot find",

        "don't have that information",
    ]

    actual_refused = any(
        phrase in actual_answer
        for phrase in refusal_phrases
    )

    # ========================================================
    # Information Exists
    #
    # Refusing would be incorrect.
    # ========================================================

    if expected_found:

        score = not actual_refused

    # ========================================================
    # Information Does Not Exist
    #
    # Refusal is required.
    # ========================================================

    else:

        score = (
            actual_refused

            and outputs.get(
                "grounded",
                False,
            )
            is False
        )

    return {

        "key":
            "refusal_correctness",

        "score":
            1.0 if score else 0.0,
    }


# ============================================================
# Run Evaluation
# ============================================================

def run_evaluation():

    client = Client()

    evaluators = [

        retrieval_correctness,

        answer_correctness,

        grounding_correctness,

        refusal_correctness,
    ]

    # ========================================================
    # Run LangSmith Experiment
    # ========================================================

    results = client.evaluate(

        target,

        data=DATASET_NAME,

        evaluators=evaluators,

        experiment_prefix=(
            "memory-vault-rag-evaluation"
        ),

        max_concurrency=1,

        metadata={

            "application":
                "Memory Vault",

            "phase":
                "Phase 12",

            "evaluation_type":
                "RAG",

            "metrics":
                (
                    "retrieval,answer,"
                    "grounding,refusal"
                ),
        },
    )

    print(
        "Evaluation completed."
    )

    print(
        f"Experiment: "
        f"{results.experiment_name}"
    )

    print(
        results
    )


if __name__ == "__main__":

    run_evaluation()