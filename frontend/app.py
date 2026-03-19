import streamlit as st, requests

API_URL = "http://localhost:8000/api"
st.set_page_config(page_title="LangChain Issue AI", page_icon="🔍", layout="wide")
st.title("🔍 LangChain Issue AI")
st.caption("LangChain GitHub 이슈 기반 트러블슈팅 어시스턴트")

query = st.text_area("에러 메시지 또는 질문", placeholder="예: RecursionError when using ConversationChain", height=100)
top_k = st.slider("참고 이슈 수", 3, 10, 5)

if st.button("검색", type="primary") and query:
    with st.spinner("검색 중..."):
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": query, "top_k": top_k}, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            st.subheader("💡 답변")
            st.markdown(data["answer"])
            st.subheader("📎 참고 이슈")
            for src in data["sources"]:
                with st.expander(f"Issue #{src['issue_number']} ({src['chunk_type']}) score: {src['score']:.3f}"):
                    (st.code if src["chunk_type"]=="code" else st.write)(src["content"])
                    st.markdown(f"[GitHub 원본]({src['issue_url']})")
        except Exception as e:
            st.error(f"오류: {e}")
