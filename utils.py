import json
import os
from typing import Any, List, Tuple

from langchain_community.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from termcolor import colored

from config import RagConfig


def load_documents_and_vector_store(config: RagConfig) -> Tuple[List[Document], FAISS]:
    """
    Load documents and return the appropriate vector store based on configuration.
    
    Two flows:
    1. If vector stores exist → load them from saved files
    2. If vector stores don't exist → create them and save them
    
    Then return the appropriate store based on config.use_text_splitter
    
    Returns:
        Tuple[List[Document], FAISS]: (documents, selected_vector_store)
    """
    try:
        if config.debug_mode:
            print(colored("\nDocument Loading", "blue"))
            print(colored("=" * 40, "blue"))

        # Check which stores exist
        original_exists = os.path.exists(config.vector_store_path)
        split_exists = os.path.exists(config.split_vector_store_path)
        
        # FLOW 1: Vector stores don't exist → CREATE them
        if not original_exists or not split_exists:
            if config.debug_mode:
                print(colored("Vector stores not found. Creating new ones...", "yellow"))
            from vector_stores import create_vector_stores
            original_store, split_store = create_vector_stores(config)
            
        # FLOW 2: Vector stores exist → LOAD them
        else:
            original_store = FAISS.load_local(
                folder_path=config.vector_store_path,
                embeddings=config.embedding_model_instance,
                allow_dangerous_deserialization=True,
            )
            
            split_store = FAISS.load_local(
                folder_path=config.split_vector_store_path,
                embeddings=config.embedding_model_instance,
                allow_dangerous_deserialization=True,
            )
            
            if config.debug_mode:
                if not original_exists or not split_exists:
                    if config.debug_mode:
                        print(colored("Vector stores not found. Creating new ones...", "yellow"))
                else:
                    original_count = len(original_store.docstore._dict)
                    split_count = len(split_store.docstore._dict)
                    print(colored(f"Loaded existing vector stores", "green"))
                    print(colored(f"  - Original: {original_count} documents", "green"))
                    print(colored(f"  - Split: {split_count} chunks", "green"))

        # Get documents from the original vector store
        documents = list(original_store.docstore._dict.values())
        
        # SELECT which store to use based on configuration
        if config.use_text_splitter:
            selected_store = split_store
            store_type = "SPLIT (chunked documents)"
        else:
            selected_store = original_store
            store_type = "ORIGINAL (full documents)"
        
        if config.debug_mode:
            print(colored(f"Using {store_type.lower()}", "cyan"))
        
        return documents, selected_store

    except Exception as e:
        print(colored(f"Error loading documents: {str(e)}", "red"))
        if config.debug_mode:
            print(colored("Troubleshooting steps:", "yellow"))
            print(colored("1. Check your internet connection for document download", "yellow"))
            print(colored("2. Verify the base_url in config.py is correct", "yellow"))
            print(colored("3. Run: python vector_stores.py to create stores manually", "yellow"))
        raise


def load_questions(questions_file: str) -> dict[str, Any]:
    """Load questions from JSON file."""
    try:
        with open(questions_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Questions file '{questions_file}' not found")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in questions file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error loading questions: {str(e)}")


def save_questions(questions_data: dict[str, Any], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(questions_data, f, indent=2, ensure_ascii=False)


def calculate_batch_stats(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        return {}

    total_time = sum(r.get("total_time", 0) for r in results)
    avg_time = total_time / len(results)

    evaluations = [r.get("evaluation", {}) for r in results if r.get("evaluation")]

    if evaluations:
        avg_accuracy = sum(e.get("accuracy_score", 0) for e in evaluations) / len(
            evaluations
        )
        avg_completeness = sum(
            e.get("completeness_score", 0) for e in evaluations
        ) / len(evaluations)

        return {
            "total_questions": len(results),
            "avg_processing_time": avg_time,
            "avg_accuracy": avg_accuracy,
            "avg_completeness": avg_completeness,
            "total_time": total_time,
        }

    return {
        "total_questions": len(results),
        "avg_processing_time": avg_time,
        "total_time": total_time,
    }

