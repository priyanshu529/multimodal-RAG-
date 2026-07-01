"""
retriever.py

Job: Build a hybrid retriever (FAISS + BM25) over the Documents, then
rerank results with FlashRank so the most relevant chunks float to the top.

This is the same retrieval pattern as your calorie tracker project,
just reused here for multimodal documents.
"""

import pickle
import os
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_huggingface import HuggingFaceEmbeddings
from flashrank import Ranker  # needed explicitly or FlashrankRerank.model_rebuild() errors

FAISS_INDEX_DIR = "data/faiss_index"
DOCS_PICKLE_PATH = "data/documents.pkl"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def add_documents_to_index(new_documents: list, index_dir: str = FAISS_INDEX_DIR, docs_path: str = DOCS_PICKLE_PATH):
    """
    Incremental ingestion: adds new_documents to the existing index without
    rebuilding from scratch. If no index exists yet, creates one.
    This is what lets you just upload a new PDF anytime, instead of
    re-running ingestion over every PDF you've ever uploaded.

    index_dir / docs_path let the caller point this at a per-chat folder
    instead of the shared default, so different chats can have separate,
    isolated document sets.
    """
    if os.path.exists(index_dir) and os.path.exists(docs_path):
        # Load what's already there
        vectorstore = FAISS.load_local(
            index_dir, embeddings, allow_dangerous_deserialization=True
        )
        with open(docs_path, "rb") as f:
            existing_documents = pickle.load(f)

        # Add the new stuff on top
        vectorstore.add_documents(new_documents)
        all_documents = existing_documents + new_documents
        
    else:
        # Nothing exists yet -- this is the first PDF ever uploaded
        vectorstore = FAISS.from_documents(new_documents, embeddings)
        all_documents = new_documents

    vectorstore.save_local(index_dir)
    os.makedirs(os.path.dirname(docs_path), exist_ok=True)
    with open(docs_path, "wb") as f:
        pickle.dump(all_documents, f)

    return len(all_documents)


def load_hybrid_retriever(k: int = 5, index_dir: str = FAISS_INDEX_DIR, docs_path: str = DOCS_PICKLE_PATH):
    """Run this every time the app starts. Loads everything from disk -- no re-embedding."""
    vectorstore = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )
    with open(docs_path, "rb") as f:
        documents = pickle.load(f)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = k

    # Combine semantic search (FAISS) with keyword search (BM25).
    # Weighted 50/50: FAISS catches "meaning", BM25 catches exact numbers/terms.
    hybrid_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    # Rerank the combined results so the truly best matches end up on top.
    reranker = FlashrankRerank(client=Ranker(), top_n=k)
    final_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid_retriever,
    )

    return final_retriever