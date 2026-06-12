import pickle
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rag_core import get_base_retriever

_hybrid_retriever = None

def get_hybrid_retriever(k=8, weights=(0.8, 0.2)):
    global _hybrid_retriever
    if _hybrid_retriever is None:
        # 读取之前保存的所有文本
        with open("all_texts.pkl", "rb") as f:
            texts = pickle.load(f)
        if not texts:
            raise ValueError("文本列表为空，请检查 all_texts.pkl 文件。")

        bm25_retriever = BM25Retriever.from_texts(texts, k=k)
        vector_retriever = get_base_retriever(k=k)

        _hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=weights
        )
    return _hybrid_retriever