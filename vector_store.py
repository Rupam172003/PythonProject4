from langchain_community.vectorstores import FAISS

def store_vectors(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
