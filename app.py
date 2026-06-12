import os
import sys
from pathlib import Path
import streamlit as st
import re
from datetime import datetime
import uuid
from hybrid_retriever import get_hybrid_retriever
from rag_core import get_llm
from gmp_compliance import compliance_check
from memory_chat import create_memory_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 检查知识库索引，若不存在则运行构建脚本
if not (Path("chroma_db").exists() and Path("all_texts.pkl").exists()):
    with st.spinner("首次运行，正在构建知识库（可能需要 5-10 分钟），请稍候..."):
        python_executable = sys.executable
        result = os.system(f'"{python_executable}" build_rag.py')
        if result != 0:
            st.error("知识库构建失败，请检查日志。")
            st.stop()
    st.success("知识库构建完成！")
    st.rerun()

st.set_page_config(page_title="智能制药助手", layout="wide")
st.title("🌿 制药行业智能助手")

st.markdown("""
<style>
.source-tag {
    background-color: #e6f7ff;
    border-left: 4px solid #1890ff;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 0.9em;
    display: inline-block;
    margin-top: 8px;
}
.source-card {
    background-color: #f9f9f9;
    border-left: 4px solid #52c41a;
    padding: 10px;
    margin: 10px 0;
    border-radius: 6px;
    font-size: 0.9em;
}
.source-card strong {
    color: #2c3e50;
}
</style>
""", unsafe_allow_html=True)

def is_date_question(question):
    q = question.strip().lower()
    exact = ["今天几号", "今天是几号", "今天日期", "今天星期几", "今天周几", "几号", "日期", "星期几", "周几", "现在几点", "现在什么时间", "几点了"]
    if q in exact:
        return True
    if re.match(r'^(今天|现在)(是|几号|日期|多少号|周几|星期几|礼拜几|几点|什么时间)[？?]?$', q):
        return True
    return False

@st.cache_resource
def load_components():
    retriever = get_hybrid_retriever()
    llm = get_llm()
    return retriever, llm

try:
    retriever, llm = load_components()
except Exception as e:
    st.error(f"加载核心组件失败: {e}")
    st.stop()

def format_docs(docs):
    if not docs:
        return "未找到相关文档。"
    return "\n\n---\n\n".join([
        f"【来源：{doc.metadata.get('source', '未知')}】\n{doc.page_content}"
        for doc in docs
    ])

with st.sidebar:
    st.header("⚙️ 功能选择")
    mode = st.radio("选择模式", ["💬 智能问答（多轮对话）", "📋 合规性检查"])
    st.markdown("---")
    st.caption("知识库包含：GMP规范、药品管理法、GMP指南、中国药典2025版、中药鉴定教材、清洁验证指南、共线生产指南、制药用水指南、变更研究指导原则")

if mode == "💬 智能问答（多轮对话）":
    st.subheader("智能问答（支持多轮对话）")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if "memory_chain" not in st.session_state:
        st.session_state.memory_chain = create_memory_chain()

    if prompt := st.chat_input("请输入问题"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if is_date_question(prompt):
                today = datetime.now().strftime("%Y年%m月%d日")
                response = f"今天是 {today}。"
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                with st.spinner("检索中..."):
                    chain = st.session_state.memory_chain
                    response = chain.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    highlighted_response = re.sub(
                        r'(【来源：.*?】)',
                        r'<span class="source-tag">\1</span>',
                        response
                    )
                    st.markdown(highlighted_response, unsafe_allow_html=True)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                docs = retriever.invoke(prompt)
                if docs:
                    with st.expander("📖 参考来源（高亮显示）"):
                        for i, doc in enumerate(docs):
                            source = doc.metadata.get('source', '未知')
                            content = doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else "")
                            st.markdown(
                                f'<div class="source-card"><strong>来源 {i + 1}：{source}</strong><br>{content}</div>',
                                unsafe_allow_html=True
                            )

else:
    st.subheader("合规性自动检查")
    st.markdown("上传文档（支持 .docx, .txt, .pdf）或直接粘贴内容，系统将基于 GMP 规范进行审核。")

    uploaded_file = st.file_uploader("选择文件", type=["docx", "txt", "pdf"])
    user_doc = ""

    if uploaded_file is not None:
        file_type = uploaded_file.name.split(".")[-1].lower()
        try:
            if file_type == "docx":
                from docx import Document
                doc = Document(uploaded_file)
                user_doc = "\n".join([para.text for para in doc.paragraphs])
            elif file_type == "txt":
                user_doc = uploaded_file.read().decode("utf-8")
            elif file_type == "pdf":
                from pypdf import PdfReader
                reader = PdfReader(uploaded_file)
                user_doc = "\n".join([page.extract_text() or "" for page in reader.pages])
            st.success(f"已加载文件：{uploaded_file.name}")
        except Exception as e:
            st.error(f"文件读取失败：{e}")

    user_doc = st.text_area("或直接粘贴文档内容", value=user_doc, height=300)

    if st.button("开始检查", type="primary"):
        if user_doc.strip():
            with st.spinner("正在比对 GMP 规范..."):
                try:
                    report = compliance_check(user_doc)
                    st.markdown("### 合规检查报告")
                    st.markdown(report)
                except Exception as e:
                    st.error(f"检查失败: {e}")
        else:
            st.warning("请上传文件或输入文档内容")