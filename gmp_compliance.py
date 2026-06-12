# gmp_compliance.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from hybrid_retriever import get_hybrid_retriever
from rag_core import get_llm

def compliance_check(user_text: str) -> str:
    """对用户提供的文本进行合规性检查，返回结构化报告"""
    llm = get_llm()
    retriever = get_hybrid_retriever(k=5)  # 检查时需要更多上下文

    def format_docs(docs):
        if not docs:
            return "未找到相关法规依据。"
        return "\n\n".join([
            f"【来源：{doc.metadata.get('source', '未知')}】\n{doc.page_content}"
            for doc in docs
        ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一位严格的GMP合规专家。请根据下列从官方规范中检索到的片段，对用户提交的文档进行分析。
请严格按照以下结构出具检查报告：

1. **潜在问题**：列出文档中可能不符合GMP要求的点。
2. **相关依据**：引用检索到的具体条款作为判断依据。
3. **修改建议**：基于依据，提出具体、可行的修改建议。
4. **风险等级**：评价为【高】/【中】/【低】。

请只基于以下【知识库片段】进行分析，不要编造内容。"""),
        ("human", "【用户文档】\n{user_text}\n\n【知识库片段】\n{context}\n\n请开始分析。")
    ])

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["user_text"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke({"user_text": user_text})