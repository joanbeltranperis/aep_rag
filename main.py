#!/usr/bin/env python3
"""
RAG Q&A System for Immunization Manual
Main execution script with configurable architecture components.
"""

import json
import os
from typing import Any
from pathlib import Path

from langchain_community.vectorstores import FAISS

from config import BLUE, GREEN, RESET, YELLOW, RagConfig
from pipeline import RAGPipeline
from vector_stores import create_vector_stores
from utils import load_documents, save_results


def load_questions(path: str) -> dict[str, Any]:
    """Load questions from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_questions(data: dict[str, Any], path: str):
    """Save questions to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def run_single_question_demo(config: RagConfig, documents: list, vector_store):
    pipeline = RAGPipeline(config)

    question = "Hola buenas tardes resulta que hoy fui a que me le pusieran la vacuna de la bcg a mi beb´e pero ella se movi´o y se le salio una gota quisiera saber si es malo que el l´ıquido se le aya salido y si es necesario volverse la a poner o si con el l´ıquido que entr´o fue suficiente estoy con el pendiente."

    human_answer = "Según el capítulo 28. Hepatitis A, …"

    print(f"\n🔄 Processing single question demo")
    print(f"{BLUE}Question: {question[:100]}...{RESET}")

    result = pipeline.process_question(
        question=question,
        documents=documents,
        vector_store=vector_store,
        human_answer=human_answer if config.enable_evaluation else None,
    )

    print(f"\n✅ Demo completed!")
    print(f"Answer length: {len(result['answer'])} characters")

    if result["evaluation"]:
        print(f"Evaluation: {result['evaluation']}")

    return result


def run_batch_evaluation(
    config: RagConfig,
    documents: list,
    vector_store,
    start_idx: int = 0,
    end_idx: int = 10,
):
    if not config.enable_evaluation:
        print(f"{YELLOW}⚠️  Evaluation is disabled in current configuration{RESET}")
        return

    pipeline = RAGPipeline(config)

    try:
        json_questions = load_questions("questions_evaluation.json")
    except FileNotFoundError:
        print(
            f"{YELLOW}⚠️  questions_evaluation.json not found, skipping batch evaluation{RESET}"
        )
        return

    questions_list = json_questions.get("questions", [])

    print(f"\n🔄 Running batch evaluation ({start_idx} to {end_idx})")

    for i in range(start_idx, min(end_idx, len(questions_list))):
        try:
            json_question = questions_list[i]
            question = json_question["question"]
            human_answer = json_question.get("human_answer", {}).get("content", "")

            print(f"\n🔄 Processing question {i + 1}/{len(questions_list)}{RESET}")

            result = pipeline.process_question(
                question=question,
                documents=documents,
                vector_store=vector_store,
                human_answer=human_answer,
            )

            # Update the question data
            json_question["rag_answer"] = {
                "content": result["answer"],
                "evaluation": result["evaluation"].get("rag_evaluation", {}),
                "metrics": {
                    "total_time": result["metrics"].total_time,
                    "retrieval_time": result["metrics"].retrieval_time,
                    "generation_time": result["metrics"].generation_time,
                    "documents_retrieved": result["metrics"].documents_retrieved,
                    "context_length": result["metrics"].context_length,
                },
            }

            if "human_answer" not in json_question:
                json_question["human_answer"] = {"content": human_answer}

            json_question["human_answer"]["evaluation"] = result["evaluation"].get(
                "human_evaluation", {}
            )

            # Save progress
            if config.save_results:
                save_questions(json_questions, "questions_evaluation.json")
                print(f"✅ Saved progress for question {i + 1}{RESET}")

        except Exception as e:
            print(f"{YELLOW}❌ Error processing question {i + 1}: {e}{RESET}")
            if "questions" in json_questions and i < len(json_questions["questions"]):
                json_questions["questions"][i]["error"] = str(e)
                if config.save_results:
                    save_questions(json_questions, "questions_evaluation.json")
            continue

    print(f"\n✅ Batch evaluation completed!")


def main():
    """Main execution function for the RAG Q&A system."""
    
    # Initialize configuration
    config = RagConfig()  # Default setup with optimal settings
    
    # Load documents and create vector store
    print("🔄 Loading documents and initializing vector store...")
    documents, vector_store = load_documents(config)
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline(config)
    
    # Example question
    question = "What are the contraindications for live vaccines?"
    
    print(f"\n🚀 Starting RAG Q&A System")
    print(f"📊 Configuration: {config.get_active_components()}")
    
    # Process the question
    result = pipeline.process(
        question=question,
        documents=documents,
        vector_store=vector_store
    )
    
    # Save results if enabled
    if config.save_results:
        output_path = Path("results") / "rag_output.json"
        save_results(result, output_path)
        print(f"\n💾 Results saved to: {output_path}")
    
    print(f"\n🎉 RAG System Processing Completed!")


if __name__ == "__main__":
    main()
