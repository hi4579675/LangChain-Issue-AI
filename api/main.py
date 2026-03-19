import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from google import genai

from collector.github_client import GitHubClient
from pipeline.vector_store import VectorStore
from pipeline.retriever import HybridRetriever
from pipeline.reranker import CrossEncoderReranker
from .routes import query, sync


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    app.state.gh       = GitHubClient(token=os.environ["GITHUB_TOKEN"])
    app.state.gemini   = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    app.state.store    = VectorStore(dsn=os.environ["DATABASE_URL"])
    app.state.retriever = HybridRetriever(conn=app.state.store.conn)
    app.state.reranker  = CrossEncoderReranker()

    yield

    app.state.store.close()  # DB 연결 종료 (retriever도 같은 conn 공유)


app = FastAPI(title="LangChain Issue AI", version="0.1.0", lifespan=lifespan)
app.include_router(query.router, prefix="/api")
app.include_router(sync.router,  prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
