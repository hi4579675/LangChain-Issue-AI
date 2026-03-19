"""DB에서 평가셋 구성 — 이슈 질문 텍스트를 query, issue_number를 정답으로"""
import random
import psycopg2
import psycopg2.extras
from dataclasses import dataclass


@dataclass
class QAPair:
    query: str
    ground_truth: int  # issue_number


def build_dataset(conn: psycopg2.extensions.connection,
                  n_samples: int = 100,
                  seed: int = 42) -> list[QAPair]:
    """
    DB에 인덱싱된 이슈 중 랜덤 샘플링.
    각 이슈의 첫 번째 question 청크를 query, issue_number를 정답으로 사용.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT issue_number FROM chunks
            WHERE chunk_type = 'text' AND is_solution = FALSE
        """)
        all_issues = [r["issue_number"] for r in cur.fetchall()]

    if not all_issues:
        raise RuntimeError("DB에 데이터가 없습니다. collect_and_index.py를 먼저 실행하세요.")

    random.seed(seed)
    sampled = random.sample(all_issues, min(n_samples, len(all_issues)))

    pairs = []
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        for issue_num in sampled:
            cur.execute("""
                SELECT content FROM chunks
                WHERE issue_number = %s AND chunk_type = 'text' AND is_solution = FALSE
                ORDER BY id LIMIT 1
            """, (issue_num,))
            row = cur.fetchone()
            if row:
                pairs.append(QAPair(query=row["content"], ground_truth=issue_num))

    print(f"평가셋 구성 완료: {len(pairs)}개 QA 쌍 (전체 {len(all_issues)}개 이슈 중 샘플링)")
    return pairs
