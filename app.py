from __future__ import annotations

import os
import re
from typing import Any

import fitz
import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_groq import ChatGroq

from tools import FINANCIAL_TOOLS

import pathlib
st.markdown("<style>"+pathlib.Path("style.css").read_text()+"</style>", unsafe_allow_html=True)
st.set_page_config(page_title="Financial Document Analyst", page_icon="📊", layout="wide")
st.markdown("### 📊 Financial Document Analyst")
st.caption("AI-powered financial analysis - Agentic RAG - Financial Tools - MCP")
st.divider()
api_key = os.getenv("GROQ_API_KEY", "")
try:
    api_key = api_key or st.secrets.get("GROQ_API_KEY", "")
except Exception:
    pass

if not api_key:
    st.warning("Set GROQ_API_KEY in Streamlit secrets or environment variables.")
    st.stop()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


@st.cache_resource(show_spinner=False)
def build_store(uploaded_files):
    docs: list[Document] = []
    for uploaded_file in uploaded_files:
        pdf = fitz.open(stream=uploaded_file.getvalue(), filetype='pdf')
        name = uploaded_file.name
        for page_num, page in enumerate(pdf, 1):
            text = page.get_text('text').strip()
            if text:
                docs.append(Document(page_content=text, metadata={'source': name, 'page': page_num}))
        pdf.close()
    if not docs:
        raise ValueError('No readable text found in PDF.')

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    emb = FastEmbedEmbeddings(model_name='BAAI/bge-small-en-v1.5')
    client = QdrantClient(location=':memory:')
    vector_size = len(emb.embed_query('test'))
    client.create_collection('docs', vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))
    store = QdrantVectorStore(client=client, collection_name='docs', embedding=emb)
    store.add_documents(chunks)
    return store, len(docs), len(chunks)

def retrieve(store, query: str, k: int = 5):
    return store.similarity_search(query, k=k)


def context_from_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', 'document')} - Page {d.metadata.get('page', '?')}] {d.page_content}" for d in docs
    )


def run_agent(llm, store, question: str, history: list[dict[str, str]]):
    retriever_tool = None

    from langchain_core.tools import StructuredTool

    def retrieve_document_context(query: str) -> str:
        """Retrieve relevant passages from the uploaded financial document."""
        docs = retrieve(store, query, k=5)
        return context_from_docs(docs)

    retriever_tool = StructuredTool.from_function(
        retrieve_document_context,
        name="search_financial_document",
        description="Search the uploaded financial PDF and return relevant passages with page numbers.",
    )

    tools = [retriever_tool, *FINANCIAL_TOOLS]
    tool_map = {t.name: t for t in tools}
    model = llm.bind_tools(tools)

    system = (
        "You are a financial document analyst. Answer using the uploaded document. "
        "Use search_financial_document whenever factual document information is needed. "
        "Use calculator/percentage_change/profit_margin/cagr for arithmetic rather than mental math. "
        "Never invent figures. If the document does not contain the answer, say so. "
        "Cite page numbers when document evidence is used."
    )

    messages: list[Any] = [HumanMessage(content=system)]
    for item in history[-6:]:
        messages.append(HumanMessage(content=item["q"]))
        messages.append(AIMessage(content=item["a"]))
    messages.append(HumanMessage(content=question))

    for _ in range(5):
        response = model.invoke(messages)
        if not response.tool_calls:
            return response.content
        messages.append(response)
        for call in response.tool_calls:
            tool = tool_map.get(call["name"])
            if not tool:
                result = f"Unknown tool: {call['name']}"
            else:
                result = tool.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return "I could not complete the tool-assisted reasoning loop."


uploaded = st.file_uploader("Upload financial PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded:
    try:
        with st.spinner("Indexing PDF..."):
            store, page_count, chunk_count = build_store(uploaded)
        st.success(f"PDFs indexed - {page_count} pages - {chunk_count} chunks")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        tab_chat, tab_compare, tab_eval, tab_arch = st.tabs(["💬 Agentic Q&A", "📊 Compare PDFs", "📈 RAG Evaluation", "🧩 Architecture"])

        with tab_chat:
            col1, col2 = st.columns([4, 1])
            with col1:
                q = st.text_input("Ask a question", placeholder="What was the revenue growth from 2024 to 2025?")
            with col2:
                if st.button("Clear chat"):
                    st.session_state.chat_history = []
                    st.rerun()

            if q:
                llm = ChatGroq(api_key=api_key, model=MODEL, temperature=0)
                with st.spinner("Agent is retrieving evidence and selecting tools..."):
                    answer = run_agent(llm, store, q, st.session_state.chat_history)
                st.session_state.chat_history.append({"q": q, "a": answer})

            for item in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(item["q"])
                with st.chat_message("assistant"):
                    st.write(item["a"])

        with tab_compare:
            st.subheader("Compare Financial Documents")
            if len(uploaded) < 2:
                st.info("Upload at least 2 PDFs to compare them.")
            else:
                compare_q = st.text_input("Comparison question", placeholder="Compare revenue and operating profit across the uploaded reports.", key="compare_q")
                if st.button("Compare documents") and compare_q:
                    docs = retrieve(store, compare_q, k=min(10, max(5, len(uploaded) * 5)))
                    grouped = {}
                    for d in docs:
                        grouped.setdefault(d.metadata.get("source", "document"), []).append(d)
                    context = "\n\n".join(
                        f"DOCUMENT: {source}\n" + "\n".join(
                            f"[Page {d.metadata.get('page', '?')}] {d.page_content}" for d in items
                        )
                        for source, items in grouped.items()
                    )
                    llm = ChatGroq(api_key=api_key, model=MODEL, temperature=0)
                    prompt = (
                        "Compare the uploaded financial documents using only the evidence below. "
                        "Clearly identify differences, similarities, and changes. "
                        "Do not invent figures. Cite document names and page numbers.\n\n"
                        f"QUESTION: {compare_q}\n\nEVIDENCE:\n{context}"
                    )
                    with st.spinner("Comparing documents..."):
                        comparison = llm.invoke([HumanMessage(content=prompt)]).content
                    st.markdown(comparison)
        with tab_eval:
            st.subheader('RAG Evaluation')
            st.caption('Evaluate retrieval quality and answer grounding.')
            eval_q = st.text_input('Evaluation question', key='eval_q')
            expected = st.text_input('Expected keywords', key='eval_expected', placeholder='150 million, revenue')
            if st.button('Run evaluation') and eval_q:
                docs = retrieve(store, eval_q, k=5)
                keywords = [x.strip().lower() for x in expected.split(',') if x.strip()]
                if not keywords:
                    st.warning('Enter expected keywords, e.g. 150 million, revenue.')
                    st.stop()
                doc_text = ' '.join(d.page_content.lower() for d in docs)
                answer = run_agent(ChatGroq(api_key=api_key, model=MODEL, temperature=0), store, eval_q, [])
                answer_text = answer.lower()

                hits = sum(1 for kw in keywords if kw in doc_text)
                answer_hits = sum(1 for kw in keywords if kw in answer_text)
                relevant_docs = sum(1 for d in docs if any(kw in d.page_content.lower() for kw in keywords)); precision_at_k = relevant_docs / len(docs) if docs else 0.0
                recall_at_k = hits / len(keywords) if keywords else 0.0
                reciprocal_rank = 1.0 if hits > 0 else 0.0
                answer_score = 1.0 if answer_hits > 0 else 0.0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric('Precision@K', f'{precision_at_k:.0%}')
                c2.metric('Recall@K', f'{recall_at_k:.0%}')
                c3.metric('MRR', f'{reciprocal_rank:.2f}')
                c4.metric('Answer Score', f'{answer_score:.0%}')

                st.write('**Generated answer:**', answer)
                st.write('**Retrieved pages:**', sorted({d.metadata.get('page') for d in docs}))

        with tab_arch:
            st.markdown("### Pipeline")
            st.code("PDF â†’ PyMuPDF â†’ Chunking â†’ BGE Embeddings â†’ Qdrant â†’ Agent â†’ RAG Tool / Financial Tools â†’ Groq LLM â†’ Answer + Sources", language="text")
            st.markdown("### Added engineering skills")
            st.markdown("- Agentic tool calling\n- Financial calculation tools\n- Multi-turn chat memory\n- Retrieval/answer evaluation\n- MCP tool server (`mcp_server.py`)\n- Docker deployment")
            st.info(f"Groq model: {MODEL}")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload a PDF to begin.")














st.divider()
st.markdown("## 📘 About this Project")
st.caption("AI-powered financial document intelligence built with Agentic RAG.")
st.markdown("### 🚀 Key Features")
a1,a2,a3,a4=st.columns(4)
with a1:
    st.info("💬 **Agentic Q&A**`nAsk questions from financial PDFs with grounded answers and source citations.")
with a2:
    st.info("📊 **Compare PDFs**`nCompare multiple financial reports and identify important financial changes.")
with a3:
    st.info("📈 **RAG Evaluation**`nEvaluate retrieval and answer quality using Precision@K, Recall@K, MRR and Answer Score.")
with a4:
    st.info("🧩 **Architecture**`nExplore the RAG pipeline, agents, financial tools, MCP and deployment.")
st.markdown("### ⚙️ Technology & Capabilities")
st.caption("Modern AI, RAG and production-oriented technology stack")
t1,t2,t3,t4,t5=st.columns(5)
t1.info("🐍 Python")
t2.info("🔗 LangChain")
t3.info("🧠 BGE + Qdrant")
t4.info("🤖 Groq LLM")
t5.info("🔌 MCP + Docker")
st.markdown("### 👨‍💻 **Built & Engineered by Tushant Singh**")
st.success("🚀 Open to AI/ML Engineer · AI Engineer · Generative AI Engineer opportunities.")


