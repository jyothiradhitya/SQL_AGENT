import numpy as np
from langchain_ollama import OllamaEmbeddings

from lms_chatbot.config import settings


def get_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_url)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.asarray(a, dtype=np.float32)
    b_arr = np.asarray(b, dtype=np.float32)

    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0

    return float(np.dot(a_arr, b_arr) / denom)


def rank_by_similarity(
    query_vector: list[float],
    candidates: dict[str, list[float]],
    top_n: int,
) -> list[tuple[str, float]]:
    scored = [
        (name, cosine_similarity(query_vector, vector))
        for name, vector in candidates.items()
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_n]
