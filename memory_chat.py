from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from hybrid_retriever import get_hybrid_retriever
from rag_core import get_llm

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def format_docs(docs):
    if not docs:
        return "未找到相关文档。"
    return "\n\n---\n\n".join([
        f"【来源：{doc.metadata.get('source', '未知')}】\n{doc.page_content}"
        for doc in docs
    ])

def create_memory_chain():
    llm = get_llm()
    retriever = get_hybrid_retriever(k=4)

    # 修改后的 prompt：禁止声明，允许补充知识
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的制药质量与中药鉴定助手。请根据以下检索到的知识库片段回答用户问题。

    回答规则：
    1. 如果问题的答案分布在多个知识库片段中，请将它们**整合**后给出**完整**的回答，不要遗漏任何一条信息。
    2. 如果【知识库片段】中包含足够回答用户问题的信息，请直接给出答案，不要添加任何前缀说明（如“根据知识库片段”）。
    3. 如果【知识库片段】中没有足够的信息来直接回答用户问题，请使用以下格式回应：
   “抱歉，根据您提供的知识库片段，没有直接找到关于「XXX」的信息。不过，根据我的专业知识可以推断：...”
   其中「XXX」为用户问题中的核心关键词。然后基于你的专业知识给出合理推断或补充答案。
    4. 严禁使用“没有找到”、“无法找到”、“没有提及”等否定表述（第3条中的“没有直接找到”除外，因为那是礼貌道歉的一部分）。
    5. 无论哪种情况，都要确保回答专业、清晰、有帮助

    【知识库片段】
    {context}"""),
        ("placeholder", "{history}"),
        ("human", "{input}")
    ])

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return conversational_chain