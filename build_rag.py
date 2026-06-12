import os
# 不再需要镜像源，因为使用本地模型
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import re
import glob
import pickle
from docx import Document
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# ... 以下 extract_text_from_xxx 函数保持不变 ...

data_dir = "data"
all_docs = []

extensions = ["*.docx", "*.txt", "*.pdf"]
for ext in extensions:
    for file_path in glob.glob(os.path.join(data_dir, ext)):
        print(f"正在加载文件: {file_path}")
        # ... 根据扩展名提取 content ...
        doc = LCDocument(page_content=content, metadata={"source": os.path.basename(file_path)})
        all_docs.append(doc)

print(f"共加载 {len(all_docs)} 个文档")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
)
documents = text_splitter.split_documents(all_docs)
print(f"切分完成，共 {len(documents)} 个文本片段")

# 使用本地模型，强制离线
embeddings = HuggingFaceEmbeddings(
    model_name="./bge-small-zh-v1.5",
    model_kwargs={'local_files_only': True}
)
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("向量库构建完成")

all_texts = [doc.page_content for doc in documents]
with open("all_texts.pkl", "wb") as f:
    pickle.dump(all_texts, f)
print("已保存 all_texts.pkl")

# 以下 RAG 链部分保持不变 ...
