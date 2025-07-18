import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

from termcolor import colored

from config import RagConfig
from pipeline import RAGPipeline
from utils import load_documents_and_vector_store, load_questions


class BatchProcessor:
    def __init__(self, config: RagConfig):
        self.config = config
        self.pipeline = RAGPipeline(config)
        self.documents, self.vector_store = load_documents_and_vector_store(config)

    def process_random_questions(
        self,
        questions_file: str = "questions.json",
        output_file: str = "batch_processing_results.json",
        num_questions: int = 10,
        delay_between_questions: float = 2.0,
    ):
        print(colored("Starting Batch Processing", "green"))
        print(f"Input file: {questions_file}")
        print(f"Output file: {output_file}")
        print(f"Target questions: {num_questions}")
        print(f"Delay between questions: {delay_between_questions}s")
        print(f"Configuration: {self.config.get_active_components()}")

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
            print(
                colored(
                    f"Requested more questions than available, processing all {num_questions}",
                    "yellow",
                )
            )

        random.seed(70)
        selected_indices = random.sample(range(len(all_questions)), num_questions)
        print(f"Selected indices: {selected_indices}")

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
        print(f"Starting processing at {datetime.now().strftime('%H:%M:%S')}")

        start_time = time.time()
        success_count = 0
        error_count = 0

        for i, question_idx in enumerate(selected_indices, 1):
            question_data = all_questions[question_idx]
            question = question_data.get("question", "")
            human_answer = question_data.get("human_answer", {}).get("content", "")

            print(
                colored(
                    f"Processing question {i}/{num_questions} (index {question_idx})",
                    "blue",
                )
            )
            print(f"Question: {question[:100]}...")

            try:
                pipeline_start = time.time()
                result = self.pipeline.process(
                    question=question,
                    documents=self.documents,
                    vector_store=self.vector_store,
                    human_answer=human_answer,
                )
                pipeline_time = time.time() - pipeline_start

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

                if "evaluation" in result and result["evaluation"]:
                    detailed_result["evaluation"] = result["evaluation"]

                results["results"].append(detailed_result)
                success_count += 1

                print(
                    colored(f"Completed successfully in {pipeline_time:.2f}s", "green")
                )

                output_path = Path(output_file)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                print(colored(f"Results saved to {output_file}", "green"))

                if i < num_questions:
                    print(f"Waiting {delay_between_questions}s before next question...")
                    time.sleep(delay_between_questions)

            except Exception as e:
                error_count += 1
                print(colored(f"Error processing question {i}: {str(e)}", "red"))

                error_data = {
                    "original_index": question_idx,
                    "question": question,
                    "error": str(e),
                    "processed_at": datetime.now().isoformat(),
                }
                results["results"].append(error_data)

                output_path = Path(output_file)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)

                print(f"Waiting {delay_between_questions}s before next question...")
                time.sleep(delay_between_questions)

                continue

        total_time = time.time() - start_time
        print(colored("Batch Processing Completed!", "green"))
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        print(f"Results saved to: {output_file}")

        results["metadata"]["completed_at"] = datetime.now().isoformat()
        results["metadata"]["successful"] = success_count
        results["metadata"]["errors"] = error_count
        results["metadata"]["total_time_minutes"] = total_time / 60

        output_path = Path(output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    """Main function for batch evaluation."""
    parser = argparse.ArgumentParser(
        description="Run batch processing on random questions"
    )
    parser.add_argument(
        "--num-questions",
        "-n",
        type=int,
        default=5,
        help="Number of questions to process (default: 5)",
    )
    parser.add_argument(
        "--input-file",
        "-i",
        default="questions.json",
        help="Input questions file (default: questions.json)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        default="batch_evaluation_results.json",
        help="Output results file (default: batch_evaluation_results.json)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=1.0,
        help="Delay between questions in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    config = RagConfig()
    config.basic_setup()

    config.enable_evaluation = True
    config.use_chapter_filtering = True
    config.use_text_splitter = True
    config.use_reranker = True
    config.log_stats = False

    processor = BatchProcessor(config)

    processor.process_random_questions(
        questions_file=args.input_file,
        output_file=args.output_file,
        num_questions=args.num_questions,
        delay_between_questions=args.delay,
    )


if __name__ == "__main__":
    main()
