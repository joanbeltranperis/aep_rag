#!/usr/bin/env python3
"""
RAG Q&A System for Immunization Manual
Main execution script with configurable architecture components.
"""

from termcolor import colored

from config import RagConfig
from pipeline import RAGPipeline
from utils import load_documents_and_vector_store


def main():
    """Main function to run the RAG pipeline."""
    # Initialize configuration
    config = RagConfig()

    # Set up configuration for desired features
    config.enable_evaluation = True  # Enable to test evaluation
    config.debug_mode = True  # Enable for detailed debugging
    config.log_stats = True  # Show performance statistics

    # Show model initialization status if debug mode is enabled
    if config.debug_mode:
        config.show_model_status()

    # Initialize pipeline
    pipeline = RAGPipeline(config)

    # Load documents and vector store
    if not config.debug_mode:
        print(colored("Loading documents and initializing pipeline...", "blue"))
    documents, vector_store = load_documents_and_vector_store(config)

    # Sample question for testing
    question = "What is the main contribution of this work?"

    # Sample human answer for evaluation testing
    human_answer = """
    The main contribution of this work is the development of a Retrieval-Augmented Generation (RAG) system 
    that combines document retrieval with language model generation to provide accurate, contextual answers. 
    The system demonstrates improved performance over baseline approaches by leveraging both semantic search 
    and generative capabilities.
    """

    if not config.debug_mode:
        print(f"\n{colored('Question:', 'yellow')}")
        print(colored("=" * 40, "yellow"))
        print(question)

    # Process the question
    results = pipeline.process(
        question=question,
        documents=documents,
        vector_store=vector_store,
        human_answer=human_answer.strip(),
    )

    # In normal mode, the answer is already printed by the pipeline
    # In debug mode, we can show additional results if needed
    if config.debug_mode:
        # Check for any errors that occurred
        has_generation_error = results.get("answer", "").startswith("Error:") or results.get("answer", "").startswith("I apologize")
        has_evaluation_error = config.enable_evaluation and results.get("evaluation", {}).get("error")
        
        if not (has_generation_error or has_evaluation_error):
            print(colored("\nPipeline execution completed successfully", "green"))
        elif has_generation_error and has_evaluation_error:
            print(colored("\nPipeline execution completed with generation and evaluation errors", "red"))
        elif has_generation_error:
            print(colored("\nPipeline execution completed with generation errors", "red"))
        elif has_evaluation_error:
            print(colored("\nPipeline execution completed with evaluation errors", "red"))


if __name__ == "__main__":
    main()
