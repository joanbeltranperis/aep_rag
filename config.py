import os
from typing import Any

import google.generativeai as genai
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Load environment variables from .env file
load_dotenv()

# Color constants for terminal output
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


class RagConfig:
    def __init__(self):
        # Core models
        self.embedding_model = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
        self.generative_model_name = "gemini-1.5-flash"
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # Feature flags
        self.use_chapter_filtering: bool = True
        self.use_reranker: bool = True
        self.enable_evaluation: bool = False
        self.debug_mode: bool = False
        self.log_stats: bool = True

        # Retrieval parameters
        self.top_k_retrieval: int = 60
        self.top_k_reranked: int = 30
        self.total_chapters: int = 53

        # Initialize model instances
        self._setup_models()

    def _setup_models(self):
        # Get API key from environment or use fallback
        api_key = os.getenv("GEMINI_API_KEY", "AIzaSyDSFs3nmrUwzpBHPMAOdlEyBEzutPu60nA")

        genai.configure(api_key=api_key)
        self.generative_model = genai.GenerativeModel(self.generative_model_name)
        self.embedding_model_instance = HuggingFaceEmbeddings(
            model_name=self.embedding_model
        )
        self.reranker_model_instance = (
            CrossEncoder(self.reranker_model_name) if self.use_reranker else None
        )

    def enable_all(self):
        self.use_chapter_filtering = True
        self.use_reranker = True
        self.enable_evaluation = True
        self.debug_mode = True
        self.log_stats = True
        self._setup_models()

    def basic_setup(self):
        self.use_chapter_filtering = False
        self.use_reranker = False
        self.enable_evaluation = False
        self.debug_mode = False
        self.log_stats = True
        self.top_k_retrieval = 20
        self._setup_models()

    def set_retrieval_size(self, retrieval_k: int, reranked_k: int):
        self.top_k_retrieval = retrieval_k
        self.top_k_reranked = reranked_k

    def get_active_components(self) -> str:
        components = []
        if self.use_chapter_filtering:
            components.append("chapter_filtering")
        if self.use_reranker:
            components.append("reranker")
        if self.enable_evaluation:
            components.append("evaluation")
        if self.debug_mode:
            components.append("debug")

        return "+".join(components) + " enabled" if components else "basic setup"


def debug_log(debug_data: dict[str, Any]) -> None:
    print(f"\n{YELLOW}🐛 Debug Information{RESET}")
    print(f"{YELLOW}" + "=" * 80 + f"{RESET}")

    print(f"\n{BLUE}📋 Question:{RESET}")
    print(f"   {debug_data['question']}")

    if "retrieved_docs" in debug_data:
        docs = debug_data["retrieved_docs"]
        print(f"\n{BLUE}📚 Retrieved Documents ({len(docs)} total):{RESET}")
        for i, doc in enumerate(docs[:6], 1):  # Show first 6 documents
            print(f"\n{BLUE}   Document #{i}:{RESET}")
            print(f"   📄 Metadata: {doc.metadata}")

            content = doc.page_content
            print("   📝 Complete Content:")
            # Show full content, not truncated
            for line in content.split("\n"):
                if line.strip():
                    print(f"      {line}")
            print(f"   📏 Length: {len(content)} characters")
            print(f"   {'-' * 60}")

    if "reranked_docs" in debug_data:
        docs = debug_data["reranked_docs"]
        print(f"\n{BLUE}🎯 Reranked Documents ({len(docs)} total):{RESET}")
        for i, doc in enumerate(docs[:5], 1):  # Show first 5 reranked
            print(f"\n{BLUE}   Document #{i} (after reranking):{RESET}")
            print(f"   📄 Metadata: {doc.metadata}")

            content = doc.page_content
            print("   📝 Complete Content:")
            # Show full content, not truncated
            for line in content.split("\n"):
                if line.strip():
                    print(f"      {line}")
            print(f"   📏 Length: {len(content)} characters")
            print(f"   {'-' * 60}")

    if "context" in debug_data:
        context = debug_data["context"]
        preview = context[:500] + "..." if len(context) > 500 else context
        print(f"\n{BLUE}📝 Final Context ({len(context):,} chars total):{RESET}")
        print("   Preview (first 500 chars):")
        for line in preview.split("\n"):
            if line.strip():
                print(f"      {line}")

    if "evaluation" in debug_data and debug_data["evaluation"]:
        print(f"\n{BLUE}📊 Evaluation Results:{RESET}")
        eval_data = debug_data["evaluation"]

        if "human_evaluation" in eval_data:
            print(f"   {GREEN}👤 Human Answer Scores:{RESET}")
            human_eval = eval_data["human_evaluation"]
            for key, value in human_eval.items():
                if key != "justification":
                    print(f"      • {key.replace('_', ' ').title()}: {value}")
            if "justification" in human_eval:
                print(f"      • Justification: {human_eval['justification']}")

        if "rag_evaluation" in eval_data:
            print(f"   {GREEN}🤖 RAG Answer Scores:{RESET}")
            rag_eval = eval_data["rag_evaluation"]
            for key, value in rag_eval.items():
                if key != "justification":
                    print(f"      • {key.replace('_', ' ').title()}: {value}")
            if "justification" in rag_eval:
                print(f"      • Justification: {rag_eval['justification']}")

    print(f"\n{YELLOW}" + "=" * 80 + f"{RESET}")
