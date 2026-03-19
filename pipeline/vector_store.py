"""pgvector 기반 벡터 저장소 — Java Repository 레이어와 동일"""
import datetime
import psycopg2, psycopg2.extras
from urllib.parse import urlparse
from .chunker import Chunk


def _parse_dsn(dsn: str) -> dict:
    """DSN URL을 psycopg2 keyword 인자로 변환 (Windows 한글 환경 인코딩 오류 우회)"""
    p = urlparse(dsn)
    return dict(host=p.hostname, port=p.port or 5432,
                dbname=p.path.lstrip("/"), user=p.username, password=p.password)

CREATE_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    issue_number INT NOT NULL,
    repo TEXT NOT NULL DEFAULT 'langchain-ai/langchain',
    chunk_type TEXT NOT NULL,
    language TEXT DEFAULT '',
    content TEXT NOT NULL,
    embedding vector(768),
    weight FLOAT DEFAULT 1.0,
    is_solution BOOLEAN DEFAULT FALSE,
    issue_created_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS chunks_unique_idx
    ON chunks (issue_number, chunk_type, content);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
"""

UPSERT_SQL = """
INSERT INTO chunks (issue_number, chunk_type, language, content, embedding, weight, is_solution, issue_created_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (issue_number, chunk_type, content) DO UPDATE
    SET embedding        = EXCLUDED.embedding,
        weight           = EXCLUDED.weight,
        is_solution      = EXCLUDED.is_solution,
        issue_created_at = EXCLUDED.issue_created_at;
"""

SEARCH_SQL = """
SELECT issue_number, chunk_type, content, weight, is_solution,
       1 - (embedding <=> %s::vector) AS vector_score
FROM chunks
ORDER BY embedding <=> %s::vector
LIMIT %s;
"""


class VectorStore:
    def __init__(self, dsn: str):
        self.conn = psycopg2.connect(**_parse_dsn(dsn))
        with self.conn.cursor() as cur:
            cur.execute(CREATE_SQL)
        self.conn.commit()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
               is_solution: bool = False,
               issue_created_at: datetime.datetime | None = None):
        """청크와 벡터를 함께 저장. 동일 (issue_number, chunk_type, content) 존재 시 덮어씀"""
        with self.conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, UPSERT_SQL,
                [(c.metadata.get("issue_number"), c.chunk_type, c.language,
                  c.content, vec, c.metadata.get("weight", 1.0), is_solution, issue_created_at)
                 for c, vec in zip(chunks, vectors)])
        self.conn.commit()

    def search(self, query_vector: list[float], top_k: int = 10) -> list[dict]:
        """코사인 유사도 기준 top_k 청크 반환"""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(SEARCH_SQL, (query_vector, query_vector, top_k))
            return [dict(row) for row in cur.fetchall()]

    def close(self):
        self.conn.close()
