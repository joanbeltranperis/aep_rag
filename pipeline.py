"""
Modular RAG Pipeline System

This module provides a modular approach to the RAG (Retrieval-Augmented Generation) system,
allowing easy activation/deactivation of different components through configuration.
"""

import json
import time
from dataclasses import dataclass
from typing import Any

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import BLUE, GREEN, RED, RESET, YELLOW, RagConfig, debug_log
from templates.prompts import (
    answer_prompt_template,
    evaluation_prompt_template,
    initial_retrieval_prompt_template,
)


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
        self, question: str, documents: list[Document], vector_store: FAISS
    ) -> tuple[list[Document], float]:
        """Retrieve relevant documents based on the question."""
        start_time = time.time()

        if self.config.use_chapter_filtering:
            # Use LLM to filter relevant chapters first
            relevant_documents = self._filter_by_chapters(question, documents)
            if relevant_documents:
                filtered_vector_store = FAISS.from_documents(
                    relevant_documents, self.config.embedding_model_instance
                )
                retrieved_docs = filtered_vector_store.similarity_search(
                    question, k=self.config.top_k_retrieval
                )
            else:
                retrieved_docs = vector_store.similarity_search(
                    question, k=self.config.top_k_retrieval
                )
        else:
            # Direct vector search without chapter filtering
            retrieved_docs = vector_store.similarity_search(
                question, k=self.config.top_k_retrieval
            )

        retrieval_time = time.time() - start_time

        if self.config.log_stats:
            print(
                f"{BLUE}📚 Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s{RESET}"
            )

        return retrieved_docs, retrieval_time

    def _filter_by_chapters(
        self, question: str, documents: list[Document]
    ) -> list[Document] | None:
        """Filter documents by relevant chapters using LLM."""
        try:
            from templates.titles import document_titles

            retrieval_prompt = ChatPromptTemplate.from_template(
                initial_retrieval_prompt_template
            )
            formatted_prompt = retrieval_prompt.format(
                document_titles=document_titles,
                user_question=question,
            )

            if self.config.log_stats:
                print(f"{BLUE}🔍 Filtering relevant chapters...{RESET}")

            response = self.config.generative_model.generate_content(formatted_prompt)
            chapter_numbers = self._parse_chapter_numbers(response.text)

            if not chapter_numbers:
                if self.config.log_stats:
                    print(
                        f"{YELLOW}⚠️  Chapter filtering failed, using all documents{RESET}"
                    )
                return None

            filtered_docs = [
                doc
                for doc in documents
                if doc.metadata.get("chapter_number") in chapter_numbers
            ]

            if self.config.log_stats:
                print(
                    f"{BLUE}📖 Filtered to {len(filtered_docs)} documents from {len(chapter_numbers)} chapters{RESET}"
                )

            return filtered_docs if filtered_docs else None

        except Exception as e:
            if self.config.log_stats:
                print(f"{YELLOW}⚠️  Chapter filtering error: {str(e)}{RESET}")
            return None

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
        """Rerank documents based on relevance to the question."""
        if not self.config.use_reranker or not self.config.reranker_model_instance:
            return documents[: self.config.top_k_reranked], 0.0

        start_time = time.time()
        
        try:
            pairs = [(question, doc.page_content) for doc in documents]
            scores = self.config.reranker_model_instance.predict(pairs)

            # Combine documents with scores and sort by relevance
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            reranked_docs = [doc for doc, _ in doc_scores[: self.config.top_k_reranked]]
            reranking_time = time.time() - start_time

            if self.config.log_stats:
                top_scores = [score for _, score in doc_scores[:5]]
                print(f"{BLUE}🔄 Top reranker scores: {top_scores}{RESET}")
                print(f"{BLUE}🎯 Reranked to top {len(reranked_docs)} documents in {reranking_time:.2f}s{RESET}")

            return reranked_docs, reranking_time

        except Exception as e:
            if self.config.log_stats:
                print(f"{YELLOW}⚠️  Reranking failed: {str(e)}{RESET}")
            return documents[: self.config.top_k_reranked], time.time() - start_time


class ContextModule:
    """Handles context preparation and filtering for generation."""

    def __init__(self, config: RagConfig):
        self.config = config

    def prepare_context(self, documents: list[Document]) -> str:
        """Prepare context string from documents."""
        context_parts: list[str] = []

        for doc in documents:
            # Prepare document text
            doc_text = f"{doc.metadata}\n{doc.page_content}\n"
            context_parts.append(doc_text)

        context = "".join(context_parts)

        if self.config.log_stats:
            print(
                f"{BLUE}📝 Prepared context: {len(context):,} characters from {len(documents)} documents{RESET}"
            )

        return context


class GenerationModule:
    """Handles answer generation using the LLM."""

    def __init__(self, config: RagConfig):
        self.config = config

    def generate_answer(self, question: str, context: str) -> tuple[str, float]:
        """Generate answer using the LLM."""
        start_time = time.time()
        
        try:
            answer_prompt = ChatPromptTemplate.from_template(answer_prompt_template)
            formatted_prompt = answer_prompt.format(context=context, question=question)

            response = self.config.generative_model.generate_content(formatted_prompt)
            answer = response.text.strip()
            generation_time = time.time() - start_time

            if self.config.log_stats:
                print(f"{BLUE}💬 Generated answer in {generation_time:.2f}s ({len(answer)} chars){RESET}")

            return answer, generation_time

        except Exception as e:
            generation_time = time.time() - start_time
            error_answer = f"Error generating answer: {str(e)}"
            
            if self.config.log_stats:
                print(f"{RED}❌ Generation failed: {str(e)}{RESET}")
            
            return error_answer, generation_time


class EvaluationModule:
    """Handles answer evaluation and validation."""

    def __init__(self, config: RagConfig):
        self.config = config

    def evaluate_answer(
        self, question: str, human_answer: str, rag_answer: str
    ) -> tuple[dict[str, Any], float]:
        """Evaluate the generated answer against human answer."""
        if not self.config.enable_evaluation:
            return {}, 0.0

        start_time = time.time()
        
        try:
            evaluation_prompt = ChatPromptTemplate.from_template(evaluation_prompt_template)
            formatted_prompt = evaluation_prompt.format(
                question=question, human_answer=human_answer, rag_answer=rag_answer
            )

            response = self.config.generative_model.generate_content(formatted_prompt)
            evaluation_result = self._parse_evaluation(response.text)
            evaluation_time = time.time() - start_time

            if self.config.log_stats and evaluation_result:
                accuracy = evaluation_result.get("accuracy_score", "N/A")
                completeness = evaluation_result.get("completeness_score", "N/A")
                print(f"{BLUE}📊 Evaluation: Accuracy={accuracy}, Completeness={completeness}{RESET}")

            return evaluation_result, evaluation_time

        except Exception as e:
            evaluation_time = time.time() - start_time
            
            if self.config.log_stats:
                print(f"{RED}❌ Evaluation failed: {str(e)}{RESET}")
            
            return {"error": str(e)}, evaluation_time

    def _parse_evaluation(self, response: str) -> dict[str, Any]:
        """Clean and extract JSON from LLM response."""
        try:
            lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
            
            # Find JSON content between lines
            json_content = []
            in_json = False
            
            for line in lines:
                if line.startswith("{") or in_json:
                    in_json = True
                    json_content.append(line)
                    if line.endswith("}"):
                        break
            
            if json_content:
                json_str = " ".join(json_content)
                return json.loads(json_str)
            
            # Fallback: try to parse the entire response as JSON
            return json.loads(response.strip())
            
        except (json.JSONDecodeError, ValueError) as e:
            return {"parse_error": f"Failed to parse evaluation: {str(e)}"}


class RAGPipeline:
    """Main RAG pipeline that orchestrates all modules."""

    def __init__(self, config: RagConfig):
        self.config = config
        self.retrieval = RetrievalModule(config)
        self.reranking = RerankingModule(config)
        self.context = ContextModule(config)
        self.generation = GenerationModule(config)
        self.evaluation = EvaluationModule(config)

        if config.log_performance:
            config.print_status()

    def process(
        self,
        question: str,
        documents: list[Document],
        vector_store: FAISS,
        human_answer: str | None = None,
    ) -> dict[str, Any]:
        """Process a question through the complete RAG pipeline."""
        start_time = time.time()
        metrics = PipelineMetrics()

        # Step 1: Retrieve documents
        retrieved_docs, retrieval_time = self.retrieval.retrieve_documents(
            question, documents, vector_store
        )
        metrics.retrieval_time = retrieval_time
        metrics.documents_retrieved = len(retrieved_docs)

        # Step 2: Rerank documents (optional)
        reranked_docs, reranking_time = self.reranking.rerank_documents(
            question, retrieved_docs
        )
        metrics.reranking_time = reranking_time
        metrics.documents_reranked = len(reranked_docs)

        # Step 3: Prepare context
        context = self.context.prepare_context(reranked_docs)
        metrics.context_length = len(context)

        # Step 4: Generate answer
        answer, generation_time = self.generation.generate_answer(question, context)
        metrics.generation_time = generation_time

        # Step 5: Evaluate answer (optional)
        evaluation_result: dict[str, Any] = {}
        evaluation_time = 0.0
        if human_answer and self.config.enable_evaluation:
            evaluation_result, evaluation_time = self.evaluation.evaluate_answer(
                question, human_answer, answer
            )
            metrics.evaluation_time = evaluation_time

        # Calculate total time
        metrics.total_time = time.time() - start_time

        # Prepare debug data
        debug_data: dict[str, Any] = {
            "question": question,
            "retrieved_docs": retrieved_docs,
            "reranked_docs": reranked_docs,
            "rag_answer": answer,
            "metrics": metrics,
        }

        if evaluation_result:
            debug_data["evaluation"] = evaluation_result

        # Always show question and answer (regardless of debug settings)
        print(f"\n{GREEN}Question:{RESET} {question}")
        print(f"{GREEN}Answer:{RESET} {answer}")
        print(f"Answer length: {len(answer)} characters")

        # Show debug information only if debug is enabled
        if self.config.debug_mode:
            debug_log(debug_data)

        # Show performance stats if enabled
        if self.config.log_stats:
            self._print_performance_stats(metrics)

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

    def _print_performance_stats(self, metrics: PipelineMetrics):
        """Print detailed performance metrics."""
        print(f"\n{GREEN}⚡ Performance Metrics{RESET}")
        print(f"{GREEN}" + "=" * 40 + f"{RESET}")
        print(f"Total Time: {metrics.total_time:.2f}s")
        print(
            f"├─ Retrieval: {metrics.retrieval_time:.2f}s ({metrics.retrieval_time / metrics.total_time * 100:.1f}%)"
        )
        if metrics.reranking_time > 0:
            print(
                f"├─ Reranking: {metrics.reranking_time:.2f}s ({metrics.reranking_time / metrics.total_time * 100:.1f}%)"
            )
        print(
            f"├─ Generation: {metrics.generation_time:.2f}s ({metrics.generation_time / metrics.total_time * 100:.1f}%)"
        )
        if metrics.evaluation_time > 0:
            print(
                f"└─ Evaluation: {metrics.evaluation_time:.2f}s ({metrics.evaluation_time / metrics.total_time * 100:.1f}%)"
            )
        print(
            f"\nDocuments: {metrics.documents_retrieved} → {metrics.documents_reranked}"
        )
        print(f"Context Length: {metrics.context_length:,} chars")

