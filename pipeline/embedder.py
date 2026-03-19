"""Gemini text-embedding-004 배치 임베딩"""
from google import genai
from .chunker import Chunk

EMBED_MODEL = "models/gemini-embedding-001"  # 출력 차원: 768
BATCH_SIZE = 100


def embed_chunks(chunks: list[Chunk], client: genai.Client) -> list[list[float]]:
    """청크 리스트를 배치로 임베딩해 벡터 리스트 반환"""
    texts = [c.content for c in chunks]
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE): # 0, 100, 200, ... 인덱스로 100개씩 슬라이싱
        result = client.models.embed_content( # API 응답의 임베딩 목록
            model=EMBED_MODEL,
            contents=texts[i:i + BATCH_SIZE],
            config={"output_dimensionality": 768},
        )
        vectors.extend([e.values for e in result.embeddings]) # 배치 결과를 누적해서 하나의 리스트로 합침
    return vectors
