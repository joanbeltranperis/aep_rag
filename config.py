import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from termcolor import colored

load_dotenv()


class RagConfig:
    def __init__(self):
        self.embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
        self.generation_model_name = "gemini-2.5-flash"
        self.evaluation_model_name = "gemini-2.5-flash"
        self.generative_model_name = self.generation_model_name
        self.reranker_model_name = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
        self.vector_store_path = "vector_store"
        self.split_vector_store_path = "split_vector_store"

        self.base_url = "https://vacunasaep.org/documentos/manual/cap-{chapter_number}"

        self.use_chapter_filtering: bool = True
        self.use_reranker: bool = True
        self.enable_evaluation: bool = False
        self.debug_mode: bool = False
        self.log_stats: bool = True
        self.use_text_splitter: bool = True

        self.top_k_retrieval: int = 60
        self.top_k_reranked: int = 30
        self.total_chapters: int = 53

        self.chunk_size: int = 1000
        self.chunk_overlap: int = 200
        self.length_function = len
        self.separators = ["\n\n", "\n", " ", ""]

        self._initialize_all_models()

    def _initialize_all_models(self):
        """Initialize all models upfront."""
        if self.debug_mode:
            print(colored("\nModel Initialization", "blue"))
            print(colored("=" * 40, "blue"))

        self._initialize_generative_models()

        self._initialize_embedding_model()

        self._initialize_reranker_model()

        if self.debug_mode:
            print(colored("All models initialized successfully", "green"))

    def _initialize_generative_models(self):
        """Initialize generative model client and both generation/evaluation models."""
        if self.debug_mode:
            print(colored("Initializing generative model client...", "blue"))

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it with your Google AI API key."
            )

        try:
            self.client = genai.Client(api_key=api_key)
            if self.debug_mode:
                print(colored("Generative model client initialized", "blue"))

            if self.debug_mode:
                print(
                    colored(
                        f"Loading generation model: {self.generation_model_name}...",
                        "blue",
                    )
                )
            self.generation_model = self.client.models.get(
                model=self.generation_model_name
            )
            if self.debug_mode:
                print(colored("Generation model loaded", "blue"))

            if self.debug_mode:
                print(
                    colored(
                        f"Loading evaluation model: {self.evaluation_model_name}...",
                        "blue",
                    )
                )
            self.evaluation_model = self.client.models.get(
                model=self.evaluation_model_name
            )
            if self.debug_mode:
                print(colored("Evaluation model loaded", "blue"))

            self.generative_model = self.generation_model

        except Exception as e:
            print(colored(f"Error initializing generative models: {str(e)}", "red"))
            raise

    def _initialize_embedding_model(self):
        """Initialize embedding model."""
        if self.debug_mode:
            print(
                colored(f"Loading embedding model '{self.embedding_model}'...", "blue")
            )
            print(
                colored(
                    "(This might take a few minutes on first run as it downloads the model)",
                    "yellow",
                )
            )

        try:
            self.embedding_model_instance = HuggingFaceEmbeddings(
                model_name=self.embedding_model, cache_folder="models"
            )
            if self.debug_mode:
                print(colored("Embedding model loaded", "blue"))

        except Exception as e:
            print(colored(f"Error loading embedding model: {str(e)}", "red"))
            if self.debug_mode:
                print(colored("Troubleshooting steps:", "yellow"))
                print(colored("1. Check your internet connection", "yellow"))
                print(colored("2. Ensure you have enough disk space", "yellow"))
                print(
                    colored(
                        "3. Try removing the 'models' directory if it exists and retry",
                        "yellow",
                    )
                )
                print(
                    colored(
                        "4. If the issue persists, try using a different embedding model",
                        "yellow",
                    )
                )
            raise

    def _initialize_reranker_model(self):
        """Initialize reranker model."""
        if self.debug_mode:
            print(
                colored(
                    f"Loading reranker model '{self.reranker_model_name}'...", "blue"
                )
            )

        try:
            self.reranker_model_instance = CrossEncoder(self.reranker_model_name)
            if self.debug_mode:
                print(colored("Reranker model loaded", "blue"))

        except Exception as e:
            print(colored(f"Error loading reranker model: {str(e)}", "red"))
            raise

    def enable_all(self):
        """Enable all advanced features."""
        self.use_chapter_filtering = True
        self.use_reranker = True
        self.enable_evaluation = True
        self.debug_mode = True
        self.log_stats = True

    def basic_setup(self):
        """Configure for basic operation without advanced features."""
        self.use_chapter_filtering = False
        self.use_reranker = False
        self.enable_evaluation = False
        self.debug_mode = False
        self.log_stats = True
        self.top_k_retrieval = 30

    def set_retrieval_size(self, retrieval_k: int, reranked_k: int):
        self.top_k_retrieval = retrieval_k
        self.top_k_reranked = reranked_k

    def get_active_components(self) -> str:
        components = []
        if self.use_text_splitter:
            components.append(f"split_store({self.chunk_size})")
        else:
            components.append("original_store")
        if self.use_chapter_filtering:
            components.append("chapter_filtering")
        if self.use_reranker:
            components.append("reranker")
        if self.enable_evaluation:
            components.append("evaluation")
        if self.debug_mode:
            components.append("debug")

        return "+".join(components) + " enabled" if components else "basic setup"

    def set_text_splitter_params(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ):
        """
        Adjust text splitter parameters for vector store creation.
        Note: You need to recreate vector stores after changing these parameters.
        """
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        if separators is not None:
            self.separators = separators

    def set_generation_model(self, model_name: str):
        """Set the generation model to use."""
        self.generation_model_name = model_name
        self.generative_model_name = model_name

        try:
            self.generation_model = self.client.models.get(model=model_name)
            self.generative_model = self.generation_model
            if self.debug_mode:
                print(colored(f"Generation model updated to: {model_name}", "green"))
        except Exception as e:
            print(colored(f"Error setting generation model: {str(e)}", "red"))
            raise

    def set_evaluation_model(self, model_name: str):
        """Set the evaluation model to use."""
        self.evaluation_model_name = model_name

        try:
            self.evaluation_model = self.client.models.get(model=model_name)
            if self.debug_mode:
                print(colored(f"Evaluation model updated to: {model_name}", "green"))
        except Exception as e:
            print(colored(f"Error setting evaluation model: {str(e)}", "red"))
            raise

    def show_model_status(self):
        """Show model initialization status (called when debug mode is enabled)."""
        print(colored("\nModel Initialization Status", "blue"))
        print(colored("=" * 50, "blue"))

        if hasattr(self, "client") and hasattr(self, "generation_model"):
            print(colored(f"✓ Generation model: {self.generation_model_name}", "green"))
        else:
            print(colored("✗ Generation model: Not initialized", "red"))

        if hasattr(self, "evaluation_model"):
            print(colored(f"✓ Evaluation model: {self.evaluation_model_name}", "green"))
        else:
            print(colored("✗ Evaluation model: Not initialized", "red"))

        if hasattr(self, "embedding_model_instance"):
            print(colored(f"✓ Embedding model: {self.embedding_model}", "green"))
        else:
            print(colored("✗ Embedding model: Not initialized", "red"))

        if hasattr(self, "reranker_model_instance"):
            print(colored(f"✓ Reranker model: {self.reranker_model_name}", "green"))
        else:
            print(colored("✗ Reranker model: Not initialized", "red"))

        print(colored("Model configuration complete", "green"))


def debug_log(debug_data: dict[str, Any]) -> None:
    print(f"\n{colored('Debug Information', 'yellow')}")
    print(colored("=" * 80, "yellow"))

    if "retrieved_docs" in debug_data:
        docs = debug_data["retrieved_docs"]
        print(f"\n{colored(f'Retrieved Documents ({len(docs)} total):', 'blue')}")
        for i, doc in enumerate(docs[:6], 1):
            print(f"\n{colored(f'Document #{i}:', 'blue')}")
            print(f"Metadata: {doc.metadata}")

            content = doc.page_content
            print("Complete Content:")

            for line in content.split("\n"):
                if line.strip():
                    print(f"{line}")
            print(f"Length: {len(content)} characters")
            print(colored(f"{'-' * 60}", "blue"))

    if "reranked_docs" in debug_data:
        docs = debug_data["reranked_docs"]
        print(f"\n{colored(f'Reranked Documents ({len(docs)} total):', 'blue')}")
        for i, doc in enumerate(docs[:5], 1):
            print(f"\n{colored(f'Document #{i} (after reranking):', 'blue')}")
            print(f"Metadata: {doc.metadata}")

            content = doc.page_content
            print("Complete Content:")

            for line in content.split("\n"):
                if line.strip():
                    print(f"{line}")
            print(f"Length: {len(content)} characters")
            print(colored(f"{'-' * 60}", "blue"))

    if "evaluation" in debug_data and debug_data["evaluation"]:
        pass

    print(colored("\n" + "=" * 80, "yellow"))
