from fastapi import APIRouter, BackgroundTasks, Request
from ..models import SyncRequest
from collector.issue_fetcher import fetch_issues, fetch_comments
from collector.cleaner import make_cleaned_issue
from pipeline.chunker import split_into_chunks
from pipeline.embedder import embed_chunks
from pipeline.vector_store import VectorStore

router = APIRouter()


@router.post("/sync")
async def sync_issues(req: SyncRequest, request: Request, background_tasks: BackgroundTasks):
    state = request.app.state
    background_tasks.add_task(_run_sync, req.label, req.max_pages, state.gh, state.gemini, state.store)
    return {"message": "동기화 시작됨", "label": req.label, "max_pages": req.max_pages}


def _run_sync(label: str, max_pages: int, gh, gemini, store: VectorStore):
    synced = 0
    for raw_issue in fetch_issues(gh, label=label, max_pages=max_pages):
        comments = fetch_comments(gh, raw_issue.number)
        cleaned  = make_cleaned_issue(raw_issue, comments)
        if not cleaned:
            continue
        for text, is_sol in [(cleaned.question, False), (cleaned.solution, True)]:
            chunks  = split_into_chunks(text, cleaned.issue_number)
            vectors = embed_chunks(chunks, gemini)
            store.upsert(chunks, vectors, is_solution=is_sol)
        synced += 1
    print(f"[sync] 완료: {synced}개 이슈 동기화")
