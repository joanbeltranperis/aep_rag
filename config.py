import os
from typing import Any

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder

# Color constants for terminal output
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


class RagConfig:
    def __init__(self):
        # Core models
        self.embedding_model = "models/embedding-001"
        self.generative_model_name = "gemini-1.5-flash"
        self.reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # Feature flags
        self.use_chapter_filtering: bool = True
        self.use_reranker: bool = True
        self.enable_evaluation: bool = False
        self.debug_mode: bool = False
        self.save_results: bool = False
        self.log_stats: bool = True

        # Retrieval parameters
        self.top_k_retrieval: int = 40
        self.top_k_reranked: int = 15
        self.total_chapters: int = 30

        # Initialize model instances
        self._setup_models()

    def _setup_models(self):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.generative_model = genai.GenerativeModel(self.generative_model_name)
        self.embedding_model_instance = GoogleGenerativeAIEmbeddings(
            model=self.embedding_model
        )
        self.reranker_model_instance = (
            CrossEncoder(self.reranker_model_name) if self.use_reranker else None
        )

    def enable_all(self):
        self.use_chapter_filtering = True
        self.use_reranker = True
        self.enable_evaluation = True
        self.debug_mode = True
        self.save_results = True
        self.log_stats = True
        self._setup_models()

    def basic_setup(self):
        self.use_chapter_filtering = False
        self.use_reranker = False
        self.enable_evaluation = False
        self.debug_mode = False
        self.save_results = False
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
    print(f"{YELLOW}" + "=" * 50 + f"{RESET}")

    print(f"\n{BLUE}📋 Question:{RESET} {debug_data['question']}")

    if "retrieved_docs" in debug_data:
        docs = debug_data["retrieved_docs"]
        print(f"\n{BLUE}📚 Retrieved Documents ({len(docs)}):{RESET}")
        for i, doc in enumerate(docs[:3], 1):
            content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  {i}. {doc.metadata} | {content}")

    if "reranked_docs" in debug_data:
        docs = debug_data["reranked_docs"]
        print(f"\n{BLUE}🎯 Reranked Documents ({len(docs)}):{RESET}")
        for i, doc in enumerate(docs[:3], 1):
            content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  {i}. {doc.metadata} | {content}")

    if "context" in debug_data:
        context = debug_data["context"]
        preview = context[:200] + "..." if len(context) > 200 else context
        print(f"\n{BLUE}📝 Context Preview:{RESET} {preview}")

    if "evaluation" in debug_data and debug_data["evaluation"]:
        print(f"\n{BLUE}📊 Evaluation Results:{RESET}")
        for key, value in debug_data["evaluation"].items():
            print(f"  • {key}: {value}")

    print(f"{YELLOW}" + "=" * 50 + f"{RESET}")
