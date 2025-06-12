import json
import os
from pathlib import Path
from typing import Any, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import RagConfig, GREEN, BLUE, RESET
from vector_stores import create_vector_stores


def load_documents(config: RagConfig) -> Tuple[list[Document], FAISS]:
    """Load documents and create vector store."""
    print(f"{BLUE}📚 Loading document store...{RESET}")
    
    if not os.path.exists("document_store"):
        print(f"{BLUE}Creating new document store...{RESET}")
        vector_store = create_vector_stores(config)
    else:
        print(f"{BLUE}Loading existing document store...{RESET}")
        vector_store = FAISS.load_local(
            "document_store",
            embeddings=config.embedding_model_instance,
            allow_dangerous_deserialization=True,
        )
    
    documents = list(vector_store.docstore._dict.values())
    print(f"{GREEN}✅ Loaded {len(documents)} documents{RESET}")
    
    return documents, vector_store


def load_questions(filename: str) -> dict[str, Any]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"questions": []}


def save_questions(questions_data: dict[str, Any], filename: str) -> None:
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, indent=2, ensure_ascii=False)


def calculate_batch_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}
    
    total_time = sum(r.get("total_time", 0) for r in results)
    avg_time = total_time / len(results)
    
    evaluations = [r.get("evaluation", {}) for r in results if r.get("evaluation")]
    
    if evaluations:
        avg_accuracy = sum(e.get("accuracy_score", 0) for e in evaluations) / len(evaluations)
        avg_completeness = sum(e.get("completeness_score", 0) for e in evaluations) / len(evaluations)
        
        return {
            "total_questions": len(results),
            "avg_processing_time": avg_time,
            "avg_accuracy": avg_accuracy,
            "avg_completeness": avg_completeness,
            "total_time": total_time
        }
    
    return {
        "total_questions": len(results),
        "avg_processing_time": avg_time,
        "total_time": total_time
    } 