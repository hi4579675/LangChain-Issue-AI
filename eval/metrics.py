"""
Hit@k, MRR, NDCG@k 측정 + 평가 실행기
실행: python -m eval.metrics
"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from dotenv import load_dotenv
from google import genai

from pipeline.vector_store import VectorStore
from pipeline.retriever import HybridRetriever
from eval.dataset import build_dataset, QAPair


@dataclass
class EvalResult:
    hit_at_3: float
    hit_at_5: float
    mrr: float
    ndcg_at_5: float
    n_samples: int

    def __str__(self):
        return (
            f"  Hit@3   : {self.hit_at_3:.3f}\n"
            f"  Hit@5   : {self.hit_at_5:.3f}\n"
            f"  MRR     : {self.mrr:.3f}\n"
            f"  NDCG@5  : {self.ndcg_at_5:.3f}\n"
            f"  샘플 수  : {self.n_samples}개"
        )


def _hit(ranked: list[int], truth: int, k: int) -> float:
    return 1.0 if truth in ranked[:k] else 0.0


def _rr(ranked: list[int], truth: int) -> float:
    for i, n in enumerate(ranked):
        if n == truth:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg(ranked: list[int], truth: int, k: int) -> float:
    for i, n in enumerate(ranked[:k]):
        if n == truth:
            return 1.0 / math.log2(i + 2)
    return 0.0


def _embed_query(text: str, gemini: genai.Client) -> list[float]:
    return gemini.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[text],
        config={"output_dimensionality": 768},
    ).embeddings[0].values


def evaluate(retriever: HybridRetriever,
             gemini: genai.Client,
             dataset: list[QAPair]) -> EvalResult:
    h3 = h5 = rr = ndcg = 0.0
    for qa in dataset:
        vec = _embed_query(qa.query, gemini)
        results = retriever.search(vec, qa.query)
        ranked = [r.issue_number for r in results]

        h3   += _hit(ranked, qa.ground_truth, 3)
        h5   += _hit(ranked, qa.ground_truth, 5)
        rr   += _rr(ranked, qa.ground_truth)
        ndcg += _ndcg(ranked, qa.ground_truth, 5)

    n = len(dataset)
    return EvalResult(h3/n, h5/n, rr/n, ndcg/n, n)


if __name__ == "__main__":
    load_dotenv()
    store   = VectorStore(dsn=os.environ["DATABASE_URL"])
    gemini  = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    retriever = HybridRetriever(store.conn, top_k=10)

    dataset = build_dataset(store.conn, n_samples=100)
    print("\n[Hybrid Retriever 평가 결과]")
    result = evaluate(retriever, gemini, dataset)
    print(result)

    store.close()
