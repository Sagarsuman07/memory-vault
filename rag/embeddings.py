from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


_embeddings = None


def get_embeddings():

    global _embeddings

    if _embeddings is None:

        _embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME
        )

    return _embeddings