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
from termcolor import colored

from config import RagConfig, debug_log
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
        self, question: str, vector_store: FAISS
    ) -> tuple[list[Document], float, str]:
        """Retrieve relevant documents based on the question."""
        start_time = time.time()

        all_docs = list(vector_store.docstore._dict.values())

        filtered_docs = None
        chapter_filter_time = 0.0
        optimized_query = question  # Por defecto, usar la pregunta original

        if self.config.use_chapter_filtering:
            if self.config.debug_mode and self.config.log_stats:
                print(colored("Filtering relevant chapters and optimizing query with LLM...", "blue"))

            filtered_docs, chapter_filter_time, optimized_query = self._filter_by_chapters_and_optimize_query(
                question, all_docs
            )

            if self.config.debug_mode and self.config.log_stats:
                if optimized_query != question:
                    print(colored(f"Original question: {question}", "yellow"))
                    print(colored(f"Optimized query: {optimized_query}", "magenta"))
                else:
                    print(colored("Query optimization: No changes made", "yellow"))

            if filtered_docs:
                filtered_vector_store = FAISS.from_documents(
                    filtered_docs, vector_store.embeddings
                )

                search_store = filtered_vector_store
                if self.config.debug_mode and self.config.log_stats:
                    print(
                        colored(
                            "Using filtered documents from selected chapters", "blue"
                        )
                    )

            else:
                search_store = vector_store
                if self.config.debug_mode and self.config.log_stats:
                    print(
                        colored("Chapter filtering failed, using all documents", "red")
                    )

        else:
            search_store = vector_store
            if self.config.debug_mode and self.config.log_stats:
                print(
                    colored("Chapter filtering disabled, using all documents", "blue")
                )

        if self.config.debug_mode and self.config.log_stats:
            print(colored("Performing vector similarity search...", "blue"))
            print(colored(f"Search query: {optimized_query}", "magenta"))

        retrieved_docs = search_store.similarity_search(
            optimized_query, k=self.config.top_k_retrieval
        )

        retrieval_time = time.time() - start_time

        if self.config.debug_mode and self.config.log_stats:
            print(
                colored(
                    f"Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s",
                    "blue",
                )
            )
            if chapter_filter_time > 0:
                print(
                    colored(
                        f"  - Chapter filtering & query optimization: {chapter_filter_time:.2f}s", "blue"
                    )
                )
                print(
                    colored(
                        f"  - Vector search: {retrieval_time - chapter_filter_time:.2f}s",
                        "blue",
                    )
                )

        return retrieved_docs, retrieval_time, optimized_query

    def _filter_by_chapters_and_optimize_query(
        self, question: str, documents: list[Document]
    ) -> tuple[list[Document] | None, float, str]:
        """Filter documents by relevant chapters and optimize query using LLM."""
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
                model=self.config.generation_model_name, contents=formatted_prompt
            )
            
            chapter_numbers, optimized_query = self._parse_simple_response(response.text, question)

            if self.config.debug_mode and self.config.log_stats:
                if chapter_numbers:
                    print(
                        colored(
                            f"LLM selected chapters: {', '.join(sorted(chapter_numbers, key=int))}",
                            "blue",
                        )
                    )

                    titles_lines = document_titles.strip().split("\n")
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
                    print(
                        colored("Chapter filtering failed, using all documents", "red")
                    )
                return None, time.time() - start_time, optimized_query

            filtered_docs = [
                doc
                for doc in documents
                if doc.metadata.get("chapter_number") in chapter_numbers
            ]

            if self.config.debug_mode and self.config.log_stats:
                print(
                    colored(
                        f"Filtered from {len(documents)} to {len(filtered_docs)} documents",
                        "blue",
                    )
                )

            return filtered_docs, time.time() - start_time, optimized_query

        except Exception as e:
            if self.config.debug_mode and self.config.log_stats:
                print(colored(f"Chapter filtering & query optimization error: {str(e)}", "red"))
            return None, time.time() - start_time, question

    def _parse_simple_response(self, response: str, original_question: str) -> tuple[set[str], str]:
        """Parse simple LLM response containing chapters and optimized query."""
        try:
            lines = response.strip().split('\n')
            chapter_numbers = set()
            optimized_query = original_question  # Fallback a la pregunta original
            
            for line in lines:
                line = line.strip()
                
                # Buscar línea de capítulos
                if line.startswith('CAPÍTULOS:'):
                    chapters_text = line.split('CAPÍTULOS:', 1)[1].strip()
                    # Remover corchetes si existen
                    chapters_text = chapters_text.strip('[]')
                    chapter_numbers = self._parse_chapter_numbers(chapters_text)
                
                # Buscar línea de query optimizada
                elif line.startswith('QUERY_OPTIMIZADA:'):
                    optimized_query = line.split('QUERY_OPTIMIZADA:', 1)[1].strip()
                    # Remover corchetes si existen
                    optimized_query = optimized_query.strip('[]')
                    
            # Si no encontramos la estructura esperada, intentar parsear como antes
            if not chapter_numbers:
                chapter_numbers = self._parse_chapter_numbers(response)
                
            return chapter_numbers, optimized_query
            
        except Exception as e:
            # En caso de error, usar el método de parsing original para los capítulos
            chapter_numbers = self._parse_chapter_numbers(response)
            return chapter_numbers, original_question



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

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

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
        for doc in documents:
            context_parts.append(f"{doc.metadata}\n{doc.page_content}")

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
                model=self.config.generation_model_name, contents=formatted_prompt
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

            return (
                f"I apologize, but I'm unable to generate an answer at this moment due to a technical issue: {str(e)}",
                generation_time,
            )


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
                model=self.config.evaluation_model_name, contents=formatted_prompt
            )

            evaluation_time = time.time() - start_time

            try:
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()

                evaluation_result = json.loads(response_text)

                if self.config.debug_mode and self.config.log_stats:
                    print(
                        colored(
                            f"Evaluation completed in {evaluation_time:.2f}s", "blue"
                        )
                    )

                return evaluation_result, evaluation_time

            except json.JSONDecodeError as e:
                return {
                    "error": f"JSON parsing failed: {str(e)}",
                }, evaluation_time

        except Exception as e:
            evaluation_result = {"error": str(e)}
            evaluation_time = 0.0

        return evaluation_result, evaluation_time


class RAGPipeline:
    """
    Main RAG pipeline that orchestrates all components.
    
    Features:
    - Document retrieval with optional chapter filtering
    - Query optimization for maximum retrieval effectiveness
    - Document reranking with cross-encoder models  
    - Context preparation and answer generation
    - Answer evaluation against human responses
    
    When chapter filtering is enabled, the pipeline automatically performs
    query optimization in the same LLM call to transform user questions into
    optimal search queries by:
    - Correcting spelling and expanding abbreviations
    - Converting questions to keyword-rich statements
    - Adding medical synonyms and related terms
    - Including concepts likely to appear in documents
    - Using precise technical terminology
    - Removing interrogative words that hurt vector search
    """

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

        if self.config.debug_mode:
            print(f"\n{colored('Step 1: Document retrieval', 'blue')}")

        retrieved_docs, retrieval_time, optimized_query = self.retrieval.retrieve_documents(
            question, vector_store
        )
        metrics.retrieval_time = retrieval_time
        metrics.documents_retrieved = len(retrieved_docs)

        if self.config.debug_mode:
            print(f"\n{colored('Step 2: Document reranking', 'blue')}")

        if self.config.use_reranker:
            # Usar la query optimizada para el reranking también para consistencia
            reranked_docs, reranking_time = self.reranking.rerank_documents(
                optimized_query, retrieved_docs
            )
        else:
            if self.config.debug_mode:
                print(colored("Reranking disabled", "yellow"))
            reranked_docs = retrieved_docs[: self.config.top_k_reranked]
            reranking_time = 0.0

        metrics.reranking_time = reranking_time
        metrics.documents_reranked = len(reranked_docs)

        if self.config.debug_mode:
            print(f"\n{colored('Step 3: Context preparation', 'blue')}")

        context = self.context.prepare_context(reranked_docs)
        metrics.context_length = len(context)

        if self.config.debug_mode and self.config.log_stats:
            print(
                colored(
                    f"Context prepared: {len(context):,} characters from {len(reranked_docs)} documents",
                    "blue",
                )
            )

        if self.config.debug_mode:
            print(f"\n{colored('Step 4: Answer generation', 'blue')}")

        try:
            # Usar la query optimizada para la generación también
            answer, generation_time = self.generation.generate_answer(optimized_query, context)
            metrics.generation_time = generation_time

        except Exception as e:
            print(colored(f"Answer generation failed: {str(e)}", "red"))
            answer = f"Error: Unable to generate answer due to: {str(e)}"
            generation_time = 0.0
            metrics.generation_time = generation_time

        if self.config.debug_mode:
            print(f"\n{colored('Step 5: Answer evaluation', 'blue')}")

        evaluation_result: dict[str, Any] = {}
        evaluation_time = 0.0

        if self.config.enable_evaluation and human_answer:
            evaluation_result, evaluation_time = self.evaluation.evaluate_answer(
                optimized_query, human_answer, answer
            )
            metrics.evaluation_time = evaluation_time

        elif self.config.enable_evaluation:
            if self.config.debug_mode:
                print(
                    colored("Evaluation enabled but no human answer provided", "yellow")
                )

        else:
            if self.config.debug_mode:
                print(colored("Evaluation disabled", "yellow"))

        metrics.total_time = time.time() - start_time

        debug_data: dict[str, Any] = {
            "question": question,
            "optimized_query": optimized_query if optimized_query != question else None,
            "retrieved_docs": retrieved_docs,
            "reranked_docs": reranked_docs,
            "context": context,
            "rag_answer": answer,
            "metrics": metrics,
        }

        if evaluation_result:
            debug_data["evaluation"] = evaluation_result

        if not self.config.debug_mode:
            # Mostrar si la pregunta fue optimizada, incluso en modo no-debug
            if optimized_query != question:
                print(f"\n{colored('Query Optimization Applied:', 'cyan')}")
                print(colored("=" * 40, "cyan"))
                print(f"{colored('Original:', 'yellow')} {question}")
                print(f"{colored('Optimized:', 'magenta')} {optimized_query}")

            print(f"\n{colored('Answer:', 'green')}")
            print(colored("=" * 40, "green"))
            print(answer)

            if self.config.enable_evaluation and evaluation_result:
                print(f"\n{colored('Evaluation:', 'blue')}")
                print(colored("=" * 40, "blue"))

                if "error" in evaluation_result:
                    print(
                        colored(
                            f"Evaluation error: {evaluation_result['error']}", "red"
                        )
                    )

                elif (
                    "human_evaluation" in evaluation_result
                    and "rag_evaluation" in evaluation_result
                ):
                    self._print_evaluation_results(evaluation_result)

        else:
            print(f"\n{colored('Question:', 'yellow')}")
            print(colored("=" * 40, "yellow"))
            print(question)

            print(f"\n{colored('Answer:', 'green')}")
            print(colored("=" * 40, "green"))
            print(answer)

            if evaluation_result:
                print(f"\n{colored('Evaluation:', 'blue')}")
                print(colored("=" * 40, "blue"))

                if "error" in evaluation_result:
                    print(
                        colored(
                            f"Evaluation error: {evaluation_result['error']}", "red"
                        )
                    )

                elif (
                    "human_evaluation" in evaluation_result
                    and "rag_evaluation" in evaluation_result
                ):
                    self._print_evaluation_results(evaluation_result)

            if self.config.log_stats:
                self._print_performance_stats(metrics)

            if self.config.debug_mode:
                debug_log(debug_data)

        return {
            "question": question,
            "optimized_query": optimized_query if optimized_query != question else None,
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

        if "human_evaluation" in evaluation and "rag_evaluation" in evaluation:
            h_scores = [
                v
                for k, v in evaluation["human_evaluation"].items()
                if k != "justification" and isinstance(v, (int, float))
            ]
            r_scores = [
                v
                for k, v in evaluation["rag_evaluation"].items()
                if k != "justification" and isinstance(v, (int, float))
            ]

            if h_scores and r_scores:
                h_avg = sum(h_scores) / len(h_scores)
                r_avg = sum(r_scores) / len(r_scores)
                print(f"\n{colored('Average Scores:', 'cyan')}")
                print(f"   Human: {h_avg:.1f}/5")
                print(f"   RAG: {r_avg:.1f}/5")
                diff = r_avg - h_avg
                comparison = (
                    "RAG better"
                    if diff > 0
                    else "Human better"
                    if diff < 0
                    else "Equal"
                )
                print(f"   Difference: {diff:+.1f} ({comparison})")

    def _print_performance_stats(self, metrics: PipelineMetrics):
        """Print detailed performance metrics."""
        print(f"\n{colored('Performance Metrics', 'green')}")
        print(colored("=" * 40, "green"))
        print(f"Total Time: {metrics.total_time:.2f}s")
        print(
            colored(
                f"├─ Retrieval: {metrics.retrieval_time:.2f}s ({metrics.retrieval_time / metrics.total_time * 100:.1f}%)",
                "green",
            )
        )
        if metrics.reranking_time > 0:
            print(
                colored(
                    f"├─ Reranking: {metrics.reranking_time:.2f}s ({metrics.reranking_time / metrics.total_time * 100:.1f}%)",
                    "green",
                )
            )
        print(
            colored(
                f"├─ Generation: {metrics.generation_time:.2f}s ({metrics.generation_time / metrics.total_time * 100:.1f}%)",
                "green",
            )
        )
        if metrics.evaluation_time > 0:
            print(
                colored(
                    f"└─ Evaluation: {metrics.evaluation_time:.2f}s ({metrics.evaluation_time / metrics.total_time * 100:.1f}%)",
                    "green",
                )
            )
        print(
            colored(
                f"\nDocuments: {metrics.documents_retrieved} retrieved → {metrics.documents_reranked} reranked",
                "green",
            )
        )
        print(f"Context Length: {metrics.context_length:,} chars")
