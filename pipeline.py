"""
Modular RAG Pipeline System

This module provides a modular approach to the RAG (Retrieval-Augmented Generation) system,
allowing easy activation/deactivation of different components through configuration.
"""

import json
import time
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from termcolor import colored

from config import RagConfig, debug_log
from templates.prompts import (
    answer_prompt_template,
    evaluation_prompt_template,
    initial_retrieval_prompt_template,
)

import demjson3


@dataclass
class PipelineMetrics:
    """Store performance metrics for the pipeline."""

    total_time: float = 0.0
    retrieval_time: float = 0.0
    reranking_time: float = 0.0
    generation_time: float = 0.0
    evaluation_time: float = 0.0
    documents_retrieved: int = 0
    documents_reranked: int = 0
    context_length: int = 0


class RetrievalModule:
    """Handles document retrieval with optional chapter filtering."""

    def __init__(self, config: RagConfig):
        self.config = config

    def retrieve_documents(
        self, question: str, vector_store: FAISS
    ) -> tuple[list[Document], float]:
        """Retrieve relevant documents based on the question."""
        start_time = time.time()

        # Get all documents from vector store for potential chapter filtering
        all_docs = list(vector_store.docstore._dict.values())
        
        # Step 1a: Optional chapter filtering
        filtered_docs = None
        chapter_filter_time = 0.0
        
        if self.config.use_chapter_filtering:
            if self.config.debug_mode and self.config.log_stats:
                print(colored("Filtering relevant chapters with LLM...", "blue"))
            
            filtered_docs, chapter_filter_time = self._filter_by_chapters(question, all_docs)
            
            if filtered_docs:
                # Create a temporary vector store with filtered documents
                filtered_vector_store = FAISS.from_documents(filtered_docs, vector_store.embeddings)
                search_store = filtered_vector_store
                if self.config.debug_mode and self.config.log_stats:
                    print(colored(f"Using filtered documents from selected chapters", "blue"))
            else:
                search_store = vector_store
                if self.config.debug_mode and self.config.log_stats:
                    print(colored("Chapter filtering failed, using all documents", "red"))
        else:
            search_store = vector_store
            if self.config.debug_mode and self.config.log_stats:
                print(colored("Chapter filtering disabled, using all documents", "blue"))

        # Step 1b: Vector similarity search
        if self.config.debug_mode and self.config.log_stats:
            print(colored("Performing vector similarity search...", "blue"))
            
        retrieved_docs = search_store.similarity_search(
            question, k=self.config.top_k_retrieval
        )

        retrieval_time = time.time() - start_time

        if self.config.debug_mode and self.config.log_stats:
            print(colored(f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s", "blue"))
            if chapter_filter_time > 0:
                print(colored(f"  - Chapter filtering: {chapter_filter_time:.2f}s", "blue"))
                print(colored(f"  - Vector search: {retrieval_time - chapter_filter_time:.2f}s", "blue"))

        return retrieved_docs, retrieval_time

    def _filter_by_chapters(
        self, question: str, documents: list[Document]
    ) -> tuple[list[Document] | None, float]:
        """Filter documents by relevant chapters using LLM."""
        start_time = time.time()
        try:
            from templates.titles import document_titles

            retrieval_prompt = ChatPromptTemplate.from_template(
                initial_retrieval_prompt_template
            )
            formatted_prompt = retrieval_prompt.format(
                document_titles=document_titles,
                user_question=question,
            )

            response = self.config.client.models.generate_content(
                model=self.config.generative_model_name,
                contents=formatted_prompt
            )
            chapter_numbers = self._parse_chapter_numbers(response.text)

            if self.config.debug_mode and self.config.log_stats:
                if chapter_numbers:
                    print(colored(f"LLM selected chapters: {', '.join(sorted(chapter_numbers, key=int))}", "blue"))
                    
                    # Show chapter titles for the selected numbers
                    titles_lines = document_titles.strip().split('\n')
                    selected_titles = []
                    for num in sorted(chapter_numbers, key=int):
                        for line in titles_lines:
                            if line.startswith(f"{num}."):
                                selected_titles.append(line.strip())
                                break
                    
                    if selected_titles:
                        print(colored("Selected chapter titles:", "blue"))
                        for title in selected_titles:
                            print(colored(f"  - {title}", "blue"))
                else:
                    print(colored("No chapters selected by LLM", "yellow"))

            if not chapter_numbers:
                if self.config.debug_mode and self.config.log_stats:
                    print(colored("Chapter filtering failed, using all documents", "red"))
                return None, time.time() - start_time

            filtered_docs = [
                doc
                for doc in documents
                if doc.metadata.get("chapter_number") in chapter_numbers
            ]

            if self.config.debug_mode and self.config.log_stats:
                print(colored(f"Filtered from {len(documents)} to {len(filtered_docs)} documents", "blue"))

            return filtered_docs, time.time() - start_time

        except Exception as e:
            if self.config.debug_mode and self.config.log_stats:
                print(colored(f"Chapter filtering error: {str(e)}", "red"))
            return None, time.time() - start_time

    def _parse_chapter_numbers(self, response: str) -> set[str]:
        """Parse and validate chapter numbers from LLM response."""
        try:
            items = response.strip().replace(" ", "").split(",")
            numbers = []
            
            for item in items:
                try:
                    num = int(item)
                    if 1 <= num <= self.config.total_chapters:
                        numbers.append(item)
                except ValueError:
                    continue
            
            return set(numbers)
        except Exception:
            return set()


class RerankingModule:
    """Handles document reranking using cross-encoder models."""

    def __init__(self, config: RagConfig):
        self.config = config

    def rerank_documents(
        self, question: str, documents: list[Document]
    ) -> tuple[list[Document], float]:
        """Rerank documents using the reranker model."""
        start_time = time.time()

        pairs = [(question, doc.page_content) for doc in documents]
        scores = self.config.reranker_model_instance.predict(pairs)

        # Sort documents by reranking scores in descending order
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Take top_k_reranked documents
        reranked_docs = [doc for doc, _ in scored_docs[: self.config.top_k_reranked]]
        reranking_time = time.time() - start_time

        if self.config.debug_mode and self.config.log_stats:
            print(
                colored(
                    f"Reranked to top {len(reranked_docs)} documents in {reranking_time:.2f}s",
                    "blue",
                )
            )

        return reranked_docs, reranking_time


class ContextModule:
    """Handles context preparation from documents."""

    def __init__(self, config: RagConfig):
        self.config = config

    def prepare_context(self, documents: list[Document]) -> str:
        """Prepare context string from selected documents."""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Get metadata
            chapter_title = doc.metadata.get("chapter_title", "Unknown Chapter")
            heading = doc.metadata.get("heading", "")
            subheading = doc.metadata.get("subheading", "")
            
            # Build header
            header = f"Fragmento {i}: {chapter_title}"
            if heading:
                header += f" - {heading}"
            if subheading:
                header += f" - {subheading}"
            
            context_parts.append(f"{header}\n{doc.page_content}")

        return "\n\n".join(context_parts)


class GenerationModule:
    """Handles answer generation using LLM."""

    def __init__(self, config: RagConfig):
        self.config = config

    def generate_answer(self, question: str, context: str) -> tuple[str, float]:
        """Generate answer using context and question."""
        start_time = time.time()

        try:
            prompt = ChatPromptTemplate.from_template(answer_prompt_template)
            formatted_prompt = prompt.format(context=context, question=question)

            response = self.config.client.models.generate_content(
                model=self.config.generative_model_name,
                contents=formatted_prompt
            )

            answer = response.text.strip()
            generation_time = time.time() - start_time

            if self.config.debug_mode and self.config.log_stats:
                print(colored(f"Generated answer in {generation_time:.2f}s", "blue"))

            return answer, generation_time
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Generation failed: {str(e)}"
            if self.config.debug_mode:
                print(colored(error_msg, "red"))
            
            # Return a user-friendly error message
            if "overloaded" in str(e).lower():
                return "I apologize, but the AI service is currently overloaded. Please try again in a few moments.", generation_time
            elif "quota" in str(e).lower() or "limit" in str(e).lower():
                return "I apologize, but the AI service quota has been exceeded. Please try again later.", generation_time
            elif "unavailable" in str(e).lower():
                return "I apologize, but the AI service is temporarily unavailable. Please try again later.", generation_time
            else:
                return f"I apologize, but I'm unable to generate an answer at this moment due to a technical issue: {str(e)}", generation_time


class EvaluationModule:
    """Handles answer evaluation against human answers."""

    def __init__(self, config: RagConfig):
        self.config = config

    def evaluate_answer(
        self, question: str, human_answer: str, rag_answer: str
    ) -> tuple[dict[str, Any], float]:
        """Evaluate the RAG answer against the human answer."""
        start_time = time.time()

        prompt = ChatPromptTemplate.from_template(evaluation_prompt_template)
        formatted_prompt = prompt.format(
            question=question,
            human_answer=human_answer,
            rag_answer=rag_answer,
        )

        try:
            response = self.config.client.models.generate_content(
                model=self.config.generative_model_name,
                contents=formatted_prompt
            )

            evaluation_time = time.time() - start_time

            try:
                # Parse JSON response
                evaluation_result = json.loads(response.text.strip())
                
                if self.config.debug_mode and self.config.log_stats:
                    print(colored(f"Evaluation completed in {evaluation_time:.2f}s", "blue"))
                
                return evaluation_result, evaluation_time
                
            except json.JSONDecodeError as e:
                # Handle JSON parsing error gracefully
                # Return empty evaluation result but don't crash
                return {"error": f"JSON parsing failed: {str(e)}"}, evaluation_time
                
        except Exception as e:
            # Handle evaluation errors gracefully - don't log here since evaluation module already logs
            evaluation_result = {"error": str(e)}
            evaluation_time = 0.0

        return evaluation_result, evaluation_time


class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""

    def __init__(self, config: RagConfig):
        self.config = config
        self.retrieval = RetrievalModule(config)
        self.reranking = RerankingModule(config)
        self.context = ContextModule(config)
        self.generation = GenerationModule(config)
        self.evaluation = EvaluationModule(config)

    def process(
        self,
        question: str,
        documents: list[Document],
        vector_store: FAISS,
        human_answer: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a question through the complete RAG pipeline.
        
        Args:
            question: The user question
            documents: All available documents (for reference only)
            vector_store: Pre-created vector store (original or split based on config)
            human_answer: Optional human answer for evaluation
            
        Returns:
            Dictionary containing the answer and metrics
        """
        start_time = time.time()
        metrics = PipelineMetrics()

        # Step 1: Retrieve documents from pre-created vector store
        if self.config.debug_mode:
            print(f"\n{colored('Step 1: Document retrieval', 'blue')}")
        retrieved_docs, retrieval_time = self.retrieval.retrieve_documents(question, vector_store)
        metrics.retrieval_time = retrieval_time
        metrics.documents_retrieved = len(retrieved_docs)

        # Step 2: Rerank documents (optional)
        if self.config.debug_mode:
            print(f"\n{colored('Step 2: Document reranking', 'blue')}")
        if self.config.use_reranker:
            reranked_docs, reranking_time = self.reranking.rerank_documents(question, retrieved_docs)
        else:
            if self.config.debug_mode:
                print(colored("Reranking disabled", "yellow"))
            reranked_docs = retrieved_docs[:self.config.top_k_reranked]
            reranking_time = 0.0
            
        metrics.reranking_time = reranking_time
        metrics.documents_reranked = len(reranked_docs)

        # Step 3: Prepare context
        if self.config.debug_mode:
            print(f"\n{colored('Step 3: Context preparation', 'blue')}")
        context = self.context.prepare_context(reranked_docs)
        metrics.context_length = len(context)
        if self.config.debug_mode and self.config.log_stats:
            print(colored(f"Context prepared: {len(context):,} characters from {len(reranked_docs)} documents", "blue"))

        # Step 4: Generate answer
        if self.config.debug_mode:
            print(f"\n{colored('Step 4: Answer generation', 'blue')}")
        
        try:
            answer, generation_time = self.generation.generate_answer(question, context)
            metrics.generation_time = generation_time
        except Exception as e:
            # Handle generation errors gracefully
            print(colored(f"Answer generation failed: {str(e)}", "red"))
            answer = f"Error: Unable to generate answer due to: {str(e)}"
            generation_time = 0.0
            metrics.generation_time = generation_time

        # Step 5: Evaluate answer (optional)
        if self.config.debug_mode:
            print(f"\n{colored('Step 5: Answer evaluation', 'blue')}")
        evaluation_result: dict[str, Any] = {}
        evaluation_time = 0.0
        if self.config.enable_evaluation and human_answer:
            try:
                evaluation_result, evaluation_time = self.evaluation.evaluate_answer(
                    question, human_answer, answer
                )
                metrics.evaluation_time = evaluation_time
            except Exception as e:
                # Handle evaluation errors gracefully - don't log here since evaluation module already logs
                evaluation_result = {"error": str(e)}
                evaluation_time = 0.0
        elif self.config.enable_evaluation:
            if self.config.debug_mode:
                print(colored("Evaluation enabled but no human answer provided", "yellow"))
        else:
            if self.config.debug_mode:
                print(colored("Evaluation disabled", "yellow"))
        
        # Calculate total time
        metrics.total_time = time.time() - start_time

        # Prepare debug data
        debug_data: dict[str, Any] = {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "reranked_docs": reranked_docs,
            "context": context,
            "rag_answer": answer,
            "metrics": metrics,
        }

        if evaluation_result:
            debug_data["evaluation"] = evaluation_result

        # Show minimal results in normal mode, detailed in debug mode
        if not self.config.debug_mode:
            # Normal mode: just essential info
            print(f"\n{colored('Answer:', 'green')}")
            print(colored("=" * 40, "green"))
            print(answer)
            
            # Show evaluation if enabled and available and successful
            if self.config.enable_evaluation and evaluation_result:
                if "error" in evaluation_result:
                    print(f"\n{colored('Evaluation:', 'blue')}")
                    print(colored("=" * 40, "blue"))
                    print(colored(f"Evaluation error: {evaluation_result['error']}", "red"))
                elif "human_evaluation" in evaluation_result and "rag_evaluation" in evaluation_result:
                    print(f"\n{colored('Evaluation:', 'blue')}")
                    print(colored("=" * 40, "blue"))
                    self._print_evaluation_results(evaluation_result)
        else:
            # Debug mode: detailed results
            
            # Show question and answer prominently
            print(f"\n{colored('Question:', 'yellow')}")
            print(colored("=" * 40, "yellow"))
            print(question)
            
            print(f"\n{colored('Answer:', 'green')}")
            print(colored("=" * 40, "green"))
            print(answer)

            # Show evaluation results if available and successful
            if evaluation_result:
                if "error" in evaluation_result:
                    print(f"\n{colored('Evaluation:', 'blue')}")
                    print(colored("=" * 40, "blue"))
                    print(colored(f"Evaluation error: {evaluation_result['error']}", "red"))
                elif "human_evaluation" in evaluation_result and "rag_evaluation" in evaluation_result:
                    print(f"\n{colored('Evaluation:', 'blue')}")
                    print(colored("=" * 40, "blue"))
                    self._print_evaluation_results(evaluation_result)

            # Show performance stats if enabled
            if self.config.log_stats:
                self._print_performance_stats(metrics)

            # Show debug information last if enabled
            if self.config.debug_mode:
                debug_log(debug_data)

        return {
            "question": question,
            "answer": answer,
            "total_time": metrics.total_time,
            "retrieval_time": metrics.retrieval_time,
            "reranking_time": metrics.reranking_time,
            "generation_time": metrics.generation_time,
            "evaluation_time": metrics.evaluation_time,
            "documents_retrieved": metrics.documents_retrieved,
            "documents_reranked": metrics.documents_reranked,
            "context_length": metrics.context_length,
            "evaluation": evaluation_result,
            "retrieved_documents": retrieved_docs,
            "reranked_documents": reranked_docs,
            "context": context,
        }

    def _print_evaluation_results(self, evaluation: dict[str, Any]):
        """Print evaluation results."""

        if "human_evaluation" in evaluation:
            print(f"\n{colored('Human Answer Scores:', 'green')}")
            human_eval = evaluation["human_evaluation"]
            for key, value in human_eval.items():
                if key != "justification":
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            if "justification" in human_eval:
                print(f"   Justification: {human_eval['justification']}")

        if "rag_evaluation" in evaluation:
            print(f"\n{colored('RAG Answer Scores:', 'green')}")
            rag_eval = evaluation["rag_evaluation"]
            for key, value in rag_eval.items():
                if key != "justification":
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            if "justification" in rag_eval:
                print(f"   Justification: {rag_eval['justification']}")

        # Calculate averages
        if "human_evaluation" in evaluation and "rag_evaluation" in evaluation:
            h_scores = [v for k, v in evaluation["human_evaluation"].items() if k != "justification" and isinstance(v, (int, float))]
            r_scores = [v for k, v in evaluation["rag_evaluation"].items() if k != "justification" and isinstance(v, (int, float))]
            
            if h_scores and r_scores:
                h_avg = sum(h_scores) / len(h_scores)
                r_avg = sum(r_scores) / len(r_scores)
                print(f"\n{colored('Average Scores:', 'cyan')}")
                print(f"   Human: {h_avg:.1f}/5")
                print(f"   RAG: {r_avg:.1f}/5")
                diff = r_avg - h_avg
                comparison = "RAG better" if diff > 0 else "Human better" if diff < 0 else "Equal"
                print(f"   Difference: {diff:+.1f} ({comparison})")

    def _print_performance_stats(self, metrics: PipelineMetrics):
        """Print detailed performance metrics."""
        print(f"\n{colored('Performance Metrics', 'green')}")
        print(colored("=" * 40, "green"))
        print(f"Total Time: {metrics.total_time:.2f}s")
        print(colored(f"├─ Retrieval: {metrics.retrieval_time:.2f}s ({metrics.retrieval_time / metrics.total_time * 100:.1f}%)", "green"))
        if metrics.reranking_time > 0:
            print(colored(f"├─ Reranking: {metrics.reranking_time:.2f}s ({metrics.reranking_time / metrics.total_time * 100:.1f}%)", "green"))
        print(colored(f"├─ Generation: {metrics.generation_time:.2f}s ({metrics.generation_time / metrics.total_time * 100:.1f}%)", "green"))
        if metrics.evaluation_time > 0:
            print(colored(f"└─ Evaluation: {metrics.evaluation_time:.2f}s ({metrics.evaluation_time / metrics.total_time * 100:.1f}%)", "green"))
        print(colored(f"\nDocuments: {metrics.documents_retrieved} retrieved → {metrics.documents_reranked} reranked", "green"))
        print(f"Context Length: {metrics.context_length:,} chars")

