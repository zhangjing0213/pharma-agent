from hybrid_retriever import get_hybrid_retriever
retriever = get_hybrid_retriever()
docs = retriever.invoke("最细粉")
for doc in docs:
    print(doc.page_content[:200])
    print("---")
