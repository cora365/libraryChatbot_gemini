import os
import sys
import streamlit as st

# --- LangChain: 최신 분리 패키지 임포트 ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Chroma (sqlite 대체용 pysqlite3 패치: Streamlit Cloud 호환) ---
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")  # noqa: E402
from langchain_chroma import Chroma  # noqa: E402


# =========================
# 환경 설정 (Secrets/Env)
# =========================
# Streamlit Secrets 에 GOOGLE_API_KEY 가 있어야 합니다.
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY가 설정되지 않았습니다. Settings → Secrets 에 추가하세요.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  # langchain_google_genai 에서 자동 참조


# =========================
# 캐시 유틸
# =========================
@st.cache_resource
def load_pdf_pages(file_path: str):
    """PDF 로드 (분할은 별도 단계에서 수행)."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # load_and_split 대신 안전하게 load만
    return pages


@st.cache_resource
def build_vectorstore(docs, persist_directory: str = "./chroma_db"):
    """문서를 청크로 나눈 뒤 Chroma 벡터DB 구축 (임베딩: Gemini)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY,
    )
    vs = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="library-regulations",
    )
    return vs


@st.cache_resource
def get_or_create_vectorstore(docs, persist_directory: str = "./chroma_db"):
    """기존 Chroma DB가 있으면 로드, 없으면 생성."""
    if os.path.exists(persist_directory):
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            collection_name="library-regulations",
        )
    return build_vectorstore(docs, persist_directory=persist_directory)


# =========================
# RAG 체인 초기화
# =========================
@st.cache_resource
def initialize_rag_chain(selected_model: str):
    # PDF 경로 (배포 환경에 맞추어 경로 확인)
    file_path = r"/mount/src/library_chatbot1/[챗봇프로그램및실습] 부경대학교 규정집.pdf"
    pages = load_pdf_pages(file_path)
    vectorstore = get_or_create_vectorstore(pages)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # 1) 이전 대화 맥락을 반영해 "독립 질문"으로 재구성하는 시스템 프롬프트
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question, rewrite the question as a "
        "standalone one that can be understood without the chat history. "
        "Do NOT answer; only rewrite if needed."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 2) 실제 Q&A 프롬프트 (근거 문맥만 사용, 모르면 모른다고 답변)
    qa_system_prompt = (
        "You are an assistant for question answering. Use ONLY the following retrieved "
        "context to answer. If you don't know, say you don't know. "
        "Answer in Korean, use honorifics, keep it precise, and include a friendly emoji.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 3) LLM (Gemini)
    llm = ChatGoogleGenerativeAI(
        google_api_key=GOOGLE_API_KEY,
        model=selected_model,  # e.g., "models/gemini-1.5-flash"
        temperature=0.3,
    )

    # 4) 체인 구성
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=contextualize_q_prompt
    )
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=qa_prompt)
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=question_answer_chain
    )
    return rag_chain


# =========================
# UI
# =========================
st.set_page_config(page_title="국립부경대 도서관 규정 Q&A", page_icon="📚")
st.header("국립부경대 도서관 규정 Q&A 챗봇 💬📚")

# Gemini 모델명은 langchain_google_genai 기준으로 "models/..." 형식 권장
model_label = st.selectbox(
    "Gemini 모델 선택",
    ("gemini-1.5-flash", "gemini-1.5-pro"),
    index=0,
)
MODEL_NAME = (
    f"models/{model_label}" if not model_label.startswith("models/") else model_label
)

rag_chain = initialize_rag_chain(MODEL_NAME)

# 대화 기록 (Streamlit 전용 메모리)
chat_history = StreamlitChatMessageHistory(key="chat_messages")

# 체인에 대화 기록 연결
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,  # history 객체 공급
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 초기 안내 메시지 (최초 1회)
if "messages_initialized" not in st.session_state:
    st.session_state["messages_initialized"] = True
    st.chat_message("ai").write(
        "국립부경대 도서관 규정에 대해 무엇이든 물어보세요! 필요한 조항을 찾아드릴게요. 🙂"
    )

# 기존 메시지 렌더
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 입력창
if user_q := st.chat_input("질문을 입력하세요"):
    st.chat_message("human").write(user_q)
    with st.chat_message("ai"):
        with st.spinner("검토 중..."):
            # 세션 ID는 동일 사용자 스레드 식별용
            config = {"configurable": {"session_id": "pknu-library"}}
            result = conversational_rag_chain.invoke({"input": user_q}, config=config)

            answer = result.get("answer", "")
            st.write(answer)

            # 참고 문서(컨텍스트) 공개
            if "context" in result:
                with st.expander("🔎 참고 문서 보기"):
                    for i, doc in enumerate(result["context"], start=1):
                        src = doc.metadata.get("source", "source not found")
                        st.markdown(f"**[{i}]** {src}")
                        st.caption(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))
