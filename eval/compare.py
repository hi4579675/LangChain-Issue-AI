"""
검색 구성별 성능 비교 — 가중치 기여도 분석
실행: python -m eval.compare

비교 항목:
  A. 벡터 유사도만 (baseline)
  B. + 키워드 매칭
  C. + solution 가중치
  D. + 최신성 점수 (full hybrid)
"""
import math
import datetime
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import psycopg2.extensions
import psycopg2.extras
from dotenv import load_dotenv
from google import genai

from pipeline.vector_store import VectorStore
from pipeline.retriever import SearchResult
from eval.dataset import build_dataset, QAPair
from eval.metrics import EvalResult, _hit, _rr, _ndcg, _embed_query


def _search_configurable(
    conn: psycopg2.extensions.connection,
    query_vector: list[float],
    query_text: str,
    top_k: int = 10,
    use_keyword: bool = True,
    use_solution: bool = True,
    use_recency: bool = True,
) -> list[SearchResult]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT issue_number, content, chunk_type, is_solution, weight, issue_created_at,
                   1 - (embedding <=> %s::vector) AS vscore,
                   CASE WHEN content ILIKE %s THEN 1.3 ELSE 1.0 END AS kw
            FROM chunks ORDER BY embedding <=> %s::vector LIMIT %s
        """, (query_vector, f"%{query_text}%", query_vector, top_k * 3))
        rows = cur.fetchall()

    now = datetime.datetime.now(datetime.timezone.utc)
    results = []
    for r in rows:
        vscore   = float(r["vscore"])
        kw       = float(r["kw"]) if use_keyword else 1.0
        sol      = 1.2 if (use_solution and r["is_solution"]) else 1.0

        if use_recency and r["issue_created_at"]:
            created = r["issue_created_at"]
            if created.tzinfo is None:
                created = created.replace(tzinfo=datetime.timezone.utc)
            age_days = (now - created).days
            recency = 0.8 + 0.2 * math.exp(-age_days / 365)
        else:
            recency = 1.0

        score = vscore * float(r["weight"]) * kw * sol * recency
        results.append(SearchResult(
            issue_number=r["issue_number"],
            content=r["content"],
            chunk_type=r["chunk_type"],
            is_solution=r["is_solution"],
            issue_url=f"https://github.com/langchain-ai/langchain/issues/{r['issue_number']}",
            score=score,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


def _evaluate_config(conn, gemini, dataset, **kwargs) -> EvalResult:
    h3 = h5 = rr = ndcg = 0.0
    for qa in dataset:
        vec = _embed_query(qa.query, gemini)
        results = _search_configurable(conn, vec, qa.query, **kwargs)
        ranked = [r.issue_number for r in results]
        h3   += _hit(ranked, qa.ground_truth, 3)
        h5   += _hit(ranked, qa.ground_truth, 5)
        rr   += _rr(ranked, qa.ground_truth)
        ndcg += _ndcg(ranked, qa.ground_truth, 5)
    n = len(dataset)
    return EvalResult(h3/n, h5/n, rr/n, ndcg/n, n)


def print_table(results: dict[str, EvalResult]):
    print("\n" + "=" * 62)
    print(f"{'구성':<30} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'NDCG@5':>8}")
    print("-" * 62)
    for name, r in results.items():
        print(f"{name:<30} {r.hit_at_3:>6.3f} {r.hit_at_5:>6.3f} {r.mrr:>6.3f} {r.ndcg_at_5:>8.3f}")
    print("=" * 62)


if __name__ == "__main__":
    load_dotenv()
    store  = VectorStore(dsn=os.environ["DATABASE_URL"])
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    dataset = build_dataset(store.conn, n_samples=100)
    print(f"\n총 {len(dataset)}개 QA 쌍으로 비교 평가 시작...\n")

    configs = {
        "A. 벡터만 (baseline)":           dict(use_keyword=False, use_solution=False, use_recency=False),
        "B. +키워드 매칭":                 dict(use_keyword=True,  use_solution=False, use_recency=False),
        "C. +solution 가중치":            dict(use_keyword=True,  use_solution=True,  use_recency=False),
        "D. +최신성 (full hybrid)":       dict(use_keyword=True,  use_solution=True,  use_recency=True),
    }

    results = {}
    for name, cfg in configs.items():
        print(f"평가 중: {name}")
        results[name] = _evaluate_config(store.conn, gemini, dataset, **cfg)

    print_table(results)
    store.close()
