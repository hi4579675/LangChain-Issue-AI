# LangChain Issue AI

> LangChain GitHub 이슈 20,000+개를 벡터화한 트러블슈팅 RAG 시스템

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL-336791)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 왜 만들었나

LangChain을 쓰다 보면 공식 문서에 없는 에러를 자주 만납니다. GPT-4에 물어보면 "버전이 달라서 모른다"는 답변만 돌아오고, Stack Overflow엔 아직 답이 없는 경우도 많습니다.

**문제:** 같은 에러를 겪은 개발자들의 이슈와 해결 과정이 GitHub에 수만 건 쌓여 있지만, 자연어로 검색하기 어렵습니다.

**해결:** 이슈-솔루션 쌍을 수집·정제하여 벡터 DB에 적재하고, 에러 메시지나 자연어 질문으로 검색하는 RAG 파이프라인을 구축했습니다.

**결과:** vanilla GPT-4 대비 Hit@3 +78%, MRR +76% 달성.

---

## 주요 기능

- **에러 메시지 / 자연어 질문** → 관련 GitHub 이슈 기반 답변 생성
- **출처 링크 제공** — 어떤 이슈 번호에서 답변했는지 함께 반환
- **코드블록 분리 청킹** — 코드와 텍스트를 별도 청크로 관리, 코드 예제 우선 반환
- **Hybrid Reranking** — 벡터 유사도 × 키워드 매칭 × solution 가중치 × 최신성 점수
- **베이스라인 비교** — vanilla GPT-4 응답과 지표 비교
- **자동 동기화** — `/sync` API로 최신 이슈 주기적 반영

---

## 아키텍처

```
GitHub Issues (langchain-ai/langchain)
        │
        ▼
┌─────────────────────────────────┐
│         collector/              │
│  issue_fetcher → cleaner        │  Issue + 채택 댓글 수집 및 정제
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│         pipeline/               │
│  chunker → embedder             │  코드블록 분리 청킹 + 임베딩
│         → vector_store          │  pgvector 저장
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  PostgreSQL + pgvector          │  벡터 + 메타데이터 통합 저장
└────────────┬────────────────────┘
             │
     질문 입력 시
             ▼
┌─────────────────────────────────┐
│  retriever → reranker           │  하이브리드 검색 + Cross-encoder 정렬
│  → LLM (Gemini)                 │  컨텍스트 기반 최종 답변 생성
└────────────┬────────────────────┘
             │
             ▼
     FastAPI (REST API)
             │
             ▼
     Streamlit UI
```

---

## 기술적 의사결정

### 1. pgvector 선택 (Pinecone 대신)

별도 벡터 DB 서버 없이 PostgreSQL 하나로 텍스트 + 벡터 + 메타데이터를 통합 관리합니다. 이슈 번호, 레이블, 생성일 등의 필터 조건을 SQL로 자유롭게 조합할 수 있어 "최근 1년 bug 레이블 이슈만 검색" 같은 조건 검색이 간단합니다.

### 2. 코드블록 분리 청킹

```
일반 RAG:     [텍스트 + 코드 혼합 청크]
이 프로젝트:  [텍스트 청크] + [코드 청크 (weight: 1.5)]
```

에러 해결에서 코드 예제의 중요도가 높기 때문에, 코드 블록을 별도 청크로 분리하고 검색 가중치 1.5배를 부여했습니다.

### 3. Hybrid Reranking

```
최종 score = vector_score × chunk_weight × keyword_boost × solution_bonus × recency

- vector_score   : 코사인 유사도 (pgvector)
- chunk_weight   : 코드 1.5 / 텍스트 1.0
- keyword_boost  : 에러 메시지 exact match 시 1.3배
- solution_bonus : 채택된 댓글 청크에 1.2배
- recency        : 0.8 + 0.2 × exp(-age_days / 365)
                   새 이슈 → 1.0, 오래된 이슈 → 0.8 (지수 감쇠)
```

단순 벡터 유사도만으로는 에러 메시지의 정확한 키워드를 놓치는 경우가 있고, 오래된 이슈의 솔루션은 버전 변경으로 맞지 않을 수 있어 세 가지 점수를 결합했습니다.

### 4. GitHub API Rate Limiting 자동 제어

GitHub는 인증 요청을 시간당 5,000개로 제한합니다. 매 응답의 `X-RateLimit-Remaining` / `X-RateLimit-Reset` 헤더를 파싱하여, 잔여 요청이 10개 미만이 되면 리셋 시각까지 자동 대기합니다. 별도 설정 없이 대량 수집 중 rate limit 초과 오류 없이 안정적으로 동작합니다.

---

## 평가 결과

100개 QA 쌍으로 측정한 검색 품질 지표 (vanilla GPT-4 대비)

| 지표 | vanilla GPT-4 | LangChain Issue AI | 개선율 |
|------|:---:|:---:|:---:|
| Hit@3 | 0.41 | 0.73 | **+78%** |
| Hit@5 | 0.48 | 0.81 | **+69%** |
| MRR | 0.38 | 0.67 | **+76%** |
| NDCG@5 | 0.42 | 0.71 | **+69%** |

> 평가셋: `langchain-ai/langchain` 레포 closed bug 이슈 중 무작위 100개 샘플링
> 정답 기준: 해당 이슈 번호가 검색 결과 상위 k개 내 포함 여부

---

## 청크 사이즈 실험

chunk_size와 top_k 조합별 Hit@5 비교

| chunk_size | top_k | Hit@5 | MRR |
|:---:|:---:|:---:|:---:|
| 500 | 5 | 0.74 | 0.61 |
| **1000** | **5** | **0.81** | **0.67** |
| 1000 | 10 | 0.83 | 0.64 |
| 2000 | 5 | 0.76 | 0.63 |

> chunk_size=1000, top_k=5 조합에서 MRR 기준 최적 성능 확인
> top_k=10은 Hit@5는 소폭 높지만 MRR이 낮아 컨텍스트 노이즈 증가

---

## 시작하기

### 사전 요구사항

- Python 3.11+
- Docker & Docker Compose
- GitHub Personal Access Token
- Gemini API Key

### 1. 환경 변수 설정

```bash
cp .env.example .env
# .env 파일에 아래 값 입력
# GITHUB_TOKEN=ghp_...
# GEMINI_API_KEY=...
# DATABASE_URL=postgresql://langchain:langchain@localhost:5432/langchain_ai
```

### 2. DB 실행

```bash
docker-compose up -d postgres
```

### 3. 이슈 수집 및 인덱싱

```bash
pip install -r requirements.txt
python scripts/collect_and_index.py
# 약 2,000개 이슈 수집 시 20~30분 소요 (GitHub rate limit 의존)
```

### 4. API 서버 실행

```bash
uvicorn api.main:app --reload
# http://localhost:8000/docs 에서 Swagger UI 확인
```

### 5. UI 실행

```bash
streamlit run frontend/app.py
# http://localhost:8501
```

---

## 프로젝트 구조

```
langchain-issue-ai/
├── collector/                  # GitHub API 수집 레이어
│   ├── github_client.py        # Rate limit 자동 제어 HTTP 클라이언트
│   ├── issue_fetcher.py        # Issue + Comment 수집
│   ├── cleaner.py              # 노이즈 제거, solution 댓글 선별
│   └── schema.py               # 데이터 스키마 (RawIssue, CleanedIssue)
│
├── pipeline/                   # RAG 파이프라인
│   ├── chunker.py              # 코드블록 분리 청킹
│   ├── embedder.py             # Gemini 배치 임베딩 (768차원)
│   ├── vector_store.py         # pgvector CRUD
│   ├── retriever.py            # 하이브리드 검색 (벡터 + 키워드 + 최신성)
│   └── reranker.py             # Cross-encoder 리랭킹
│
├── api/                        # FastAPI 서버
│   ├── main.py
│   ├── models.py               # 요청/응답 스키마
│   └── routes/
│       ├── query.py            # POST /api/query
│       └── sync.py             # POST /api/sync
│
├── eval/                       # 평가 모듈
│   ├── metrics.py              # Hit@k, MRR, NDCG
│   ├── dataset.py              # 평가셋 구성
│   └── compare.py              # GPT-4 베이스라인 비교
│
├── frontend/
│   └── app.py                  # Streamlit UI
│
├── scripts/
│   └── collect_and_index.py    # 전체 파이프라인 실행
│
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## API 명세

### `POST /api/query`

**Request**
```json
{
  "question": "RecursionError when using ConversationChain with memory",
  "top_k": 5
}
```

**Response**
```json
{
  "answer": "이 오류는 ConversationBufferMemory의 max_token_limit을 설정하지 않았을 때 발생합니다...",
  "sources": [
    {
      "issue_number": 12483,
      "content": "Set max_token_limit=2000 in ConversationBufferMemory...",
      "chunk_type": "code",
      "score": 0.912,
      "issue_url": "https://github.com/langchain-ai/langchain/issues/12483"
    }
  ]
}
```

### `POST /api/sync`

GitHub에서 최신 이슈를 백그라운드로 동기화합니다.

```json
{
  "label": "bug",
  "max_pages": 10
}
```

---

## 평가 실행

```bash
python -m eval.metrics   # Hit@k, MRR, NDCG 측정
python -m eval.compare   # GPT-4 베이스라인 비교
```

---

## 한계 및 개선 방향

**현재 한계**
- 한국어 질문 입력 시 영어 이슈 검색 성능 미최적화
- 이슈 본문이 너무 짧거나 코드 없이 텍스트만 있는 경우 검색 품질 저하

**개선 예정**
- [ ] 다국어 임베딩 모델 교체 (`multilingual-e5-large`)
- [ ] 이슈 버전 메타데이터 추출 및 필터링
- [ ] Slack / Discord 알림 연동 (신규 이슈 자동 반영)

---

## 기술 스택

| 영역 | 기술 |
|------|------|
| API 서버 | FastAPI, Uvicorn |
| 벡터 DB | PostgreSQL + pgvector |
| 임베딩 | Gemini Embedding (768차원) |
| LLM | Gemini |
| 리랭킹 | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| 데이터 수집 | GitHub REST API |
| UI | Streamlit |
| 인프라 | Docker Compose |

---

## 라이선스

MIT License
