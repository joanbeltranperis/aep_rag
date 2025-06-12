#!/usr/bin/env python3

import json
import random
import time
from datetime import datetime
from pathlib import Path

from config import BLUE, GREEN, RED, RESET, YELLOW, RagConfig
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
        print(f"{GREEN}🔄 Starting Batch Evaluation{RESET}")
        print(f"📁 Input file: {questions_file}")
        print(f"📄 Output file: {output_file}")
        print(f"🎯 Target questions: {num_questions}")
        print(f"⏱️  Delay between questions: {delay_between_questions}s")
        print(f"📊 Configuration: {self.config.get_active_components()}")

        # Load questions
        try:
            questions_data = load_questions(questions_file)
            all_questions = questions_data.get("questions", [])
            print(f"📚 Loaded {len(all_questions)} total questions")
        except Exception as e:
            print(f"{RED}❌ Error loading questions: {e}{RESET}")
            return

        if len(all_questions) == 0:
            print(f"{RED}❌ No questions found in {questions_file}{RESET}")
            return

        if num_questions > len(all_questions):
            num_questions = len(all_questions)
            print(
                f"{YELLOW}⚠️  Requested more questions than available, processing all {num_questions}{RESET}"
            )

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

        print(f"🎲 Selected {num_questions} random questions to process")
        print(f"⏰ Starting evaluation at {datetime.now().strftime('%H:%M:%S')}")

        # Process each question
        start_time = time.time()
        success_count = 0
        error_count = 0

        for i, question_idx in enumerate(selected_indices, 1):
            question_data = all_questions[question_idx]
            question = question_data.get("question", "")
            human_answer = question_data.get("human_answer", {}).get("content", "")

            print(
                f"\n{BLUE}📝 Processing question {i}/{num_questions} (index {question_idx}){RESET}"
            )
            print(f"❓ Question: {question[:100]}...")

            try:
                # Run the pipeline
                question_start = time.time()
                result = self.pipeline.process(
                    question=question,
                    documents=self.documents,
                    vector_store=self.vector_store,
                    human_answer=human_answer if human_answer else None,
                )
                question_time = time.time() - question_start

                # Prepare result data
                result_data = {
                    "original_index": question_idx,
                    "question": question,
                    "human_answer": human_answer,
                    "rag_answer": result["answer"],
                    "evaluation": result.get("evaluation", {}),
                    "metrics": {
                        "total_time": result["total_time"],
                        "retrieval_time": result["retrieval_time"],
                        "reranking_time": result["reranking_time"],
                        "generation_time": result["generation_time"],
                        "evaluation_time": result["evaluation_time"],
                        "documents_retrieved": result["documents_retrieved"],
                        "documents_reranked": result["documents_reranked"],
                        "context_length": result["context_length"],
                    },
                    "processed_at": datetime.now().isoformat(),
                }

                # Add to results
                results["results"].append(result_data)
                success_count += 1

                # Save after each question (incremental saving)
                output_path = Path(output_file)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                print(
                    f"{GREEN}✅ Question {i} completed in {question_time:.2f}s{RESET}"
                )

                if result.get("evaluation"):
                    eval_data = result["evaluation"]
                    if (
                        "human_evaluation" in eval_data
                        and "rag_evaluation" in eval_data
                    ):
                        h_avg = (
                            sum(
                                [
                                    v
                                    for k, v in eval_data["human_evaluation"].items()
                                    if k != "justification"
                                ]
                            )
                            / 5
                        )
                        r_avg = (
                            sum(
                                [
                                    v
                                    for k, v in eval_data["rag_evaluation"].items()
                                    if k != "justification"
                                ]
                            )
                            / 5
                        )
                        print(f"📊 Scores - Human: {h_avg:.1f}/5, RAG: {r_avg:.1f}/5")

                # Add delay between questions to avoid rate limiting
                print(f"⏳ Waiting {delay_between_questions}s before next question...")
                time.sleep(delay_between_questions)

            except Exception as e:
                error_count += 1
                print(f"{RED}❌ Error processing question {i}: {str(e)}{RESET}")

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
                print(f"⏳ Waiting {delay_between_questions}s before next question...")
                time.sleep(delay_between_questions)

                continue

        # Final summary
        total_time = time.time() - start_time
        print(f"\n{GREEN}🎉 Batch Evaluation Completed!{RESET}")
        print(f"⏱️  Total time: {total_time / 60:.1f} minutes")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Errors: {error_count}")
        print(f"📁 Results saved to: {output_file}")

        # Update metadata
        results["metadata"]["completed_at"] = datetime.now().isoformat()
        results["metadata"]["successful"] = success_count
        results["metadata"]["errors"] = error_count
        results["metadata"]["total_time_minutes"] = total_time / 60

        # Final save with metadata
        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    # Setup configuration for batch evaluation
    config = RagConfig()
    config.enable_evaluation = True  # Enable evaluation for batch processing
    config.debug_mode = False  # Disable debug to reduce output noise
    config.log_stats = True  # Disable stats to reduce output noise

    evaluator = BatchEvaluator(config)

    # Run batch evaluation
    evaluator.evaluate_random_questions(
        questions_file="questions.json",
        output_file="batch_evaluation_results.json",
        num_questions=30,  # Process 20 random questions
        delay_between_questions=2.0,  # 3 second delay to avoid rate limits
    )


if __name__ == "__main__":
    main()

