import os

from dotenv import load_dotenv
from langsmith import Client


# ============================================================
# Load .env
# ============================================================

load_dotenv()


# ============================================================
# Configuration
# ============================================================

DATASET_NAME = (
    "Memory Vault RAG Evaluation V1"
)


# ============================================================
# Validate LangSmith configuration
# ============================================================

def validate_langsmith_config():

    api_key = os.getenv(
        "LANGSMITH_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "LANGSMITH_API_KEY is not configured.\n"
            "Please add your LangSmith API key to the .env file."
        )

    if api_key == "your_langsmith_api_key":
        raise ValueError(
            "LANGSMITH_API_KEY is still using the placeholder value.\n"
            "Please replace it with your actual LangSmith API key."
        )

    if not api_key.startswith("lsv2_"):
        raise ValueError(
            "LANGSMITH_API_KEY does not look like a valid "
            "LangSmith API key.\n"
            "Check the key in your .env file."
        )

    return api_key


# ============================================================
# Evaluation Dataset
# ============================================================

def get_evaluation_examples():

    return [

        # ====================================================
        # Memory 01 — Flight
        # ====================================================

        {
            "inputs": {
                "memory_id": "memory_01",

                "memory_text": (
                    "Flight 6E123 is booked from "
                    "Hyderabad to Delhi. "
                    "Departure date is 15 September 2026. "
                    "Airline is IndiGo."
                ),

                "question": (
                    "What is the flight number?"
                ),
            },

            "outputs": {
                "expected_answer": "6E123",
                "expected_found": True,
                "expected_evidence": "6E123",
                "forbidden_evidence": "",
            },

            "metadata": {
                "memory_id": "memory_01",
                "category": "retrieval_and_answer",
            },
        },


        {
            "inputs": {
                "memory_id": "memory_01",

                "memory_text": (
                    "Flight 6E123 is booked from "
                    "Hyderabad to Delhi. "
                    "Departure date is 15 September 2026. "
                    "Airline is IndiGo."
                ),

                "question": (
                    "Where is the flight going?"
                ),
            },

            "outputs": {
                "expected_answer": "Delhi",
                "expected_found": True,
                "expected_evidence": "Delhi",
                "forbidden_evidence": "",
            },

            "metadata": {
                "memory_id": "memory_01",
                "category": "retrieval_and_answer",
            },
        },


        # ====================================================
        # Memory 01 — Not Found
        # ====================================================

        {
            "inputs": {
                "memory_id": "memory_01",

                "memory_text": (
                    "Flight 6E123 is booked from "
                    "Hyderabad to Delhi. "
                    "Departure date is 15 September 2026. "
                    "Airline is IndiGo."
                ),

                "question": (
                    "What is the hotel?"
                ),
            },

            "outputs": {
                "expected_answer": "NOT_FOUND",
                "expected_found": False,
                "expected_evidence": "",
                "forbidden_evidence": "Hotel ABC",
            },

            "metadata": {
                "memory_id": "memory_01",
                "category": "refusal",
            },
        },


        # ====================================================
        # Memory 02 — Hotel
        # ====================================================

        {
            "inputs": {
                "memory_id": "memory_02",

                "memory_text": (
                    "Hotel ABC is booked in Delhi. "
                    "The room costs 5000 rupees per night. "
                    "Check-in is 15 September 2026."
                ),

                "question": (
                    "What hotel did I book?"
                ),
            },

            "outputs": {
                "expected_answer": "Hotel ABC",
                "expected_found": True,
                "expected_evidence": "Hotel ABC",
                "forbidden_evidence": "",
            },

            "metadata": {
                "memory_id": "memory_02",
                "category": "retrieval_and_answer",
            },
        },


        {
            "inputs": {
                "memory_id": "memory_02",

                "memory_text": (
                    "Hotel ABC is booked in Delhi. "
                    "The room costs 5000 rupees per night. "
                    "Check-in is 15 September 2026."
                ),

                "question": (
                    "How much does the hotel cost per night?"
                ),
            },

            "outputs": {
                "expected_answer": "5000 rupees per night",
                "expected_found": True,
                "expected_evidence": "5000 rupees per night",
                "forbidden_evidence": "",
            },

            "metadata": {
                "memory_id": "memory_02",
                "category": "retrieval_and_answer",
            },
        },


        # ====================================================
        # Memory 02 — Not Found
        # ====================================================

        {
            "inputs": {
                "memory_id": "memory_02",

                "memory_text": (
                    "Hotel ABC is booked in Delhi. "
                    "The room costs 5000 rupees per night. "
                    "Check-in is 15 September 2026."
                ),

                "question": (
                    "What is the flight number?"
                ),
            },

            "outputs": {
                "expected_answer": "NOT_FOUND",
                "expected_found": False,
                "expected_evidence": "",
                "forbidden_evidence": "6E123",
            },

            "metadata": {
                "memory_id": "memory_02",
                "category": "refusal",
            },
        },


        # ====================================================
        # Memory 03 — Product
        # ====================================================

        {
            "inputs": {
                "memory_id": "memory_03",

                "memory_text": (
                    "The saved product is an iPhone 16. "
                    "The listed price is 70000 rupees. "
                    "The storage capacity is 256 GB."
                ),

                "question": (
                    "What product did I save?"
                ),
            },

            "outputs": {
                "expected_answer": "iPhone 16",
                "expected_found": True,
                "expected_evidence": "iPhone 16",
                "forbidden_evidence": "",
            },

            "metadata": {
                "memory_id": "memory_03",
                "category": "retrieval_and_answer",
            },
        },


        {
            "inputs": {
                "memory_id": "memory_03",

                "memory_text": (
                    "The saved product is an iPhone 16. "
                    "The listed price is 70000 rupees. "
                    "The storage capacity is 256 GB."
                ),

                "question": (
                    "What is the storage capacity?"
                ),
            },

            "outputs": {
                "expected_answer": "256 GB",
                "expected_found": True,
                "expected_evidence": "256 GB",
                "forbidden_evidence": "",
            },

            "metadata": {
                "memory_id": "memory_03",
                "category": "retrieval_and_answer",
            },
        },


        # ====================================================
        # Memory 03 — Not Found
        # ====================================================

        {
            "inputs": {
                "memory_id": "memory_03",

                "memory_text": (
                    "The saved product is an iPhone 16. "
                    "The listed price is 70000 rupees. "
                    "The storage capacity is 256 GB."
                ),

                "question": (
                    "What is the flight number?"
                ),
            },

            "outputs": {
                "expected_answer": "NOT_FOUND",
                "expected_found": False,
                "expected_evidence": "",
                "forbidden_evidence": "6E123",
            },

            "metadata": {
                "memory_id": "memory_03",
                "category": "refusal",
            },
        },
    ]


# ============================================================
# Create Dataset
# ============================================================

def create_evaluation_dataset():

    api_key = validate_langsmith_config()

    client = Client(
        api_key=api_key
    )

    # ========================================================
    # Check whether dataset already exists
    # ========================================================

    if client.has_dataset(
        dataset_name=DATASET_NAME
    ):

        print(
            f"Dataset already exists: {DATASET_NAME}"
        )

        return

    # ========================================================
    # Create Dataset
    # ========================================================

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,

        description=(
            "Memory Vault evaluation dataset "
            "for retrieval, answer correctness, "
            "grounding, and refusal behavior."
        ),
    )

    # ========================================================
    # Add Examples
    # ========================================================

    client.create_examples(
        dataset_id=dataset.id,
        examples=get_evaluation_examples(),
    )

    print()
    print(
        "Created LangSmith dataset:"
    )

    print(
        dataset.name
    )

    print()
    print(
        "Number of examples:",
        len(get_evaluation_examples())
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    create_evaluation_dataset()