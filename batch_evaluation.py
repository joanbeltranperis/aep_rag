#!/usr/bin/env python3

import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from termcolor import colored

from config import RagConfig
from pipeline import RAGPipeline
from utils import load_documents, load_questions


class BatchEvaluator:
    def __init__(self, config: RagConfig):
        self.config = config
        self.pipeline = RAGPipeline(config)
        self.documents, self.vector_store = load_documents(config)

    def evaluate_random_questions(
        self,
        questions_file: str = "questions.json",
        output_file: str = "batch_evaluation_results.json",
        num_questions: int = 10,
        delay_between_questions: float = 2.0,
    ):
        print(colored("Starting Batch Evaluation", "green"))
        print(f"Input file: {questions_file}")
        print(f"Output file: {output_file}")
        print(f"Target questions: {num_questions}")
        print(f"Delay between questions: {delay_between_questions}s")
        print(f"Configuration: {self.config.get_active_components()}")

        # Load questions
        try:
            questions_data = load_questions(questions_file)
            all_questions = questions_data.get("questions", [])
            print(f"Loaded {len(all_questions)} total questions")
        except Exception as e:
            print(colored(f"Error loading questions: {e}", "red"))
            return

        if len(all_questions) == 0:
            print(colored(f"No questions found in {questions_file}", "red"))
            return

        if num_questions > len(all_questions):
            num_questions = len(all_questions)
            print(colored(f"Requested more questions than available, processing all {num_questions}", "yellow"))

        # Select random questions
        selected_indices = random.sample(range(len(all_questions)), num_questions)

        # Initialize results
        results = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "config": self.config.get_active_components(),
                "total_questions": num_questions,
                "source_file": questions_file,
            },
            "results": [],
        }

        print(f"Selected {num_questions} random questions to process")
        print(f"Starting evaluation at {datetime.now().strftime('%H:%M:%S')}")

        # Process each question
        start_time = time.time()
        success_count = 0
        error_count = 0

        for i, question_idx in enumerate(selected_indices, 1):
            question_data = all_questions[question_idx]
            question = question_data.get("question", "")
            human_answer = question_data.get("human_answer", {}).get("content", "")

            print(colored(f"Processing question {i}/{num_questions} (index {question_idx})", "blue"))
            print(f"Question: {question[:100]}...")

            try:
                # Process through pipeline
                pipeline_start = time.time()
                result = self.pipeline.process(
                    question=question,
                    documents=self.documents,
                    vector_store=self.vector_store,
                    human_answer=human_answer,
                )
                pipeline_time = time.time() - pipeline_start

                # Prepare detailed result
                detailed_result = {
                    "original_index": question_idx,
                    "question": question,
                    "human_answer": human_answer,
                    "rag_answer": result.get("answer", ""),
                    "pipeline_time": pipeline_time,
                    "retrieval_time": result.get("retrieval_time", 0),
                    "reranking_time": result.get("reranking_time", 0),
                    "generation_time": result.get("generation_time", 0),
                    "evaluation_time": result.get("evaluation_time", 0),
                    "total_time": result.get("total_time", 0),
                    "context_length": result.get("context_length", 0),
                    "documents_retrieved": result.get("documents_retrieved", 0),
                    "documents_reranked": result.get("documents_reranked", 0),
                    "processed_at": datetime.now().isoformat(),
                }

                # Add evaluation results if available
                if "evaluation" in result and result["evaluation"]:
                    detailed_result["evaluation"] = result["evaluation"]

                results["results"].append(detailed_result)
                success_count += 1

                print(colored(f"Completed successfully in {pipeline_time:.2f}s", "green"))

                # Save intermediate results
                output_path = Path(output_file)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                print(colored(f"Results saved to {output_file}", "green"))

                # Add delay between questions (except for the last one)
                if i < num_questions:
                    print(f"Waiting {delay_between_questions}s before next question...")
                    time.sleep(delay_between_questions)

            except Exception as e:
                error_count += 1
                print(colored(f"Error processing question {i}: {str(e)}", "red"))

                # Save error info
                error_data = {
                    "original_index": question_idx,
                    "question": question,
                    "error": str(e),
                    "processed_at": datetime.now().isoformat(),
                }
                results["results"].append(error_data)

                # Save even errors
                output_path = Path(output_file)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                # Also add delay after errors to avoid rate limiting
                print(f"Waiting {delay_between_questions}s before next question...")
                time.sleep(delay_between_questions)

                continue

        # Final summary
        total_time = time.time() - start_time
        print(colored("Batch Evaluation Completed!", "green"))
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Results saved to: {output_file}")

        # Update metadata
        results["metadata"]["completed_at"] = datetime.now().isoformat()
        results["metadata"]["successful"] = success_count
        results["metadata"]["errors"] = error_count
        results["metadata"]["total_time_minutes"] = total_time / 60

        # Final save with metadata
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def analyze_results(self, results_file: str = "batch_evaluation_results.json"):
        """Analyze results from a batch evaluation."""
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            print(colored("Batch Evaluation Analysis", "green"))
            print("=" * 60)
            
            metadata = data.get("metadata", {})
            results = data.get("results", [])
            
            # Basic statistics
            total_questions = metadata.get("total_questions", len(results))
            successful = metadata.get("successful", 0)
            errors = metadata.get("errors", 0)
            total_time = metadata.get("total_time_minutes", 0)
            
            print(f"Total questions: {total_questions}")
            print(f"Successful: {successful}")
            print(f"Errors: {errors}")
            print(f"Success rate: {(successful/total_questions)*100:.1f}%")
            print(f"Total time: {total_time:.1f} minutes")
            print(f"Average time per question: {(total_time*60/total_questions):.1f}s")
            
            # Performance metrics
            if successful > 0:
                valid_results = [r for r in results if "error" not in r]
                if valid_results:
                    avg_retrieval = sum(r.get("retrieval_time", 0) for r in valid_results) / len(valid_results)
                    avg_generation = sum(r.get("generation_time", 0) for r in valid_results) / len(valid_results)
                    avg_total = sum(r.get("total_time", 0) for r in valid_results) / len(valid_results)
                    
                    print(f"\nAverage Performance:")
                    print(f"Retrieval: {avg_retrieval:.2f}s")
                    print(f"Generation: {avg_generation:.2f}s")
                    print(f"Total: {avg_total:.2f}s")
                    
                    # Document metrics
                    avg_docs_retrieved = sum(r.get("documents_retrieved", 0) for r in valid_results) / len(valid_results)
                    avg_docs_reranked = sum(r.get("documents_reranked", 0) for r in valid_results) / len(valid_results)
                    avg_context_length = sum(r.get("context_length", 0) for r in valid_results) / len(valid_results)
                    
                    print(f"\nDocument Metrics:")
                    print(f"Documents retrieved: {avg_docs_retrieved:.1f}")
                    print(f"Documents reranked: {avg_docs_reranked:.1f}")
                    print(f"Context length: {avg_context_length:,.0f} chars")
            
        except FileNotFoundError:
            print(colored(f"Results file '{results_file}' not found", "red"))
        except json.JSONDecodeError:
            print(colored(f"Invalid JSON in results file '{results_file}'", "red"))
        except Exception as e:
            print(colored(f"Error analyzing results: {str(e)}", "red"))


def main():
    """Main function for batch evaluation."""
    config = RagConfig()
    
    # Set up for evaluation mode
    config.enable_evaluation = True
    config.log_stats = False  # Reduce verbosity for batch processing
    
    evaluator = BatchEvaluator(config)
    
    # Run batch evaluation
    evaluator.evaluate_random_questions(
        num_questions=5,  # Adjust as needed
        delay_between_questions=1.0,
    )
    
    # Analyze results
    evaluator.analyze_results()


if __name__ == "__main__":
    main()

