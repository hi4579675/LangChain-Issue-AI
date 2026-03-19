from fastapi import APIRouter, Request
from ..models import QueryRequest, QueryResponse, SourceChunk

router = APIRouter()

SYSTEM_PROMPT = """You are a LangChain troubleshooting assistant.
Answer the question based only on the GitHub issue context provided below.
Be concise and always mention the relevant issue numbers."""


@router.post("/query", response_model=QueryResponse)
async def query_issues(req: QueryRequest, request: Request):
    state = request.app.state

    # 1. 질문 임베딩
    query_vector = state.gemini.models.embed_content(
        model="models/gemini-embedding-001",
        contents=[req.question],
        config={"output_dimensionality": 768},
    ).embeddings[0].values

    # 2. 하이브리드 검색
    candidates = state.retriever.search(query_vector, req.question)

    # 3. 리랭킹
    results = state.reranker.rerank(req.question, candidates, top_n=req.top_k)

    if not results:
        return QueryResponse(answer="관련 이슈를 찾지 못했습니다.", sources=[])

    # 4. LLM 답변 생성
    context = "\n\n".join(
        f"[Issue #{r.issue_number}]\n{r.content}" for r in results
    )
    llm_response = state.gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {req.question}",
    )

    return QueryResponse(
        answer=llm_response.text,
        sources=[
            SourceChunk(
                issue_number=r.issue_number,
                content=r.content,
                chunk_type=r.chunk_type,
                score=round(r.score, 4),
                issue_url=r.issue_url,
            )
            for r in results
        ],
    )
