import os
from pprint import pprint

import google.generativeai as genai
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

os.environ["GOOGLE_API_KEY"] = "AIzaSyCIfd-QrhcRqrsTE1CosgAftdt_6Kskvm8"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


class RagConfig:
    """Simple, modular RAG configuration. Enable/disable components as needed."""
    
    def __init__(self):
        # =============================================================
        # MAIN TOGGLES - Enable/disable pipeline components
        # =============================================================
        self.use_chapter_filtering = True      # Use LLM to filter relevant chapters first
        self.use_reranker = True              # Apply reranking to retrieved documents  
        self.enable_evaluation = False        # Evaluate generated answers
        self.enable_debug = False             # Show detailed debug information
        
        # =============================================================
        # RETRIEVAL SETTINGS
        # =============================================================
        self.top_k_retrieval = 40            # Number of documents to retrieve initially
        self.top_k_reranked = 15             # Number of documents after reranking
        self.max_context_length = 8000       # Maximum context length for generation
        
        # =============================================================
        # LOGGING CONTROLS
        # =============================================================
        self.log_stats = True                # Log basic retrieval statistics  
        self.log_performance = False         # Log detailed timing metrics
        self.save_results = False            # Save evaluation results to file
        
        # =============================================================
        # MODELS - Change these to experiment with different models
        # =============================================================
        self.llm_model = "gemini-2.0-flash-lite"
        self.embedding_model = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"  
        self.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        
        # =============================================================
        # DATA SETTINGS
        # =============================================================
        self.total_chapters = 53
        self.base_url = r"https://vacunasaep.org/documentos/manual/cap-{chapter_number}"
        
        # Initialize models
        self._setup_models()
    
    def _setup_models(self):
        """Initialize the models based on current configuration."""
        self.generative_model = genai.GenerativeModel(self.llm_model)
        self.embedding_model_instance = HuggingFaceEmbeddings(model_name=self.embedding_model)
        
        # Only load reranker if we're using it
        if self.use_reranker:
            self.reranker_model_instance = CrossEncoder(self.reranker_model)
        else:
            self.reranker_model_instance = None
    
    def toggle_reranker(self, enabled: bool = None):
        """Enable or disable the reranker. If no value provided, toggles current state."""
        if enabled is None:
            self.use_reranker = not self.use_reranker
        else:
            self.use_reranker = enabled
        
        # Reinitialize models
        self._setup_models()
        
        if self.log_stats:
            status = "enabled" if self.use_reranker else "disabled"
            print(f"🔄 Reranker {status}")
    
    def toggle_evaluation(self, enabled: bool = None):
        """Enable or disable evaluation. If no value provided, toggles current state."""
        if enabled is None:
            self.enable_evaluation = not self.enable_evaluation
        else:
            self.enable_evaluation = enabled
            
        if self.log_stats:
            status = "enabled" if self.enable_evaluation else "disabled"  
            print(f"📊 Evaluation {status}")
    
    def toggle_debug(self, enabled: bool = None):
        """Enable or disable debug logging. If no value provided, toggles current state."""
        if enabled is None:
            self.enable_debug = not self.enable_debug
        else:
            self.enable_debug = enabled
            
        if self.log_stats:
            status = "enabled" if self.enable_debug else "disabled"
            print(f"🐛 Debug logging {status}")
    
    def set_retrieval_size(self, initial: int, reranked: int | None = None):
        """Set retrieval parameters."""
        self.top_k_retrieval = initial
        if reranked is not None:
            self.top_k_reranked = reranked
        elif reranked is None and self.use_reranker:
            # Default reranked to half of initial if not specified
            self.top_k_reranked = initial // 2
            
        if self.log_stats:
            if self.use_reranker:
                print(f"📚 Retrieval: {self.top_k_retrieval} → {self.top_k_reranked} documents")
            else:
                print(f"📚 Retrieval: {self.top_k_retrieval} documents")
    
    def enable_all(self):
        """Enable all features for full functionality."""
        self.use_chapter_filtering = True
        self.use_reranker = True
        self.enable_evaluation = True
        self.enable_debug = True
        self.log_performance = True
        self.save_results = True
        self.top_k_retrieval = 40
        self.top_k_reranked = 15
        self._setup_models()
        
        if self.log_stats:
            print("🔥 All features enabled")
    
    def basic_setup(self):
        """Basic setup with minimal components for speed."""
        self.use_chapter_filtering = False
        self.use_reranker = False
        self.enable_evaluation = False
        self.enable_debug = False
        self.log_performance = False
        self.save_results = False
        self.top_k_retrieval = 20
        self._setup_models()
        
        if self.log_stats:
            print("⚡ Basic setup enabled")
    
    def print_status(self):
        """Print current configuration status."""
        print(f"\n{GREEN}📋 Current Configuration{RESET}")
        print(f"{GREEN}" + "=" * 40 + f"{RESET}")
        
        # Pipeline components
        print(f"{BLUE}Pipeline Components:{RESET}")
        print(f"  Chapter Filtering: {'✅' if self.use_chapter_filtering else '❌'}")
        print(f"  Reranker: {'✅' if self.use_reranker else '❌'}")
        print(f"  Evaluation: {'✅' if self.enable_evaluation else '❌'}")
        
        # Settings
        print(f"\n{BLUE}Settings:{RESET}")
        print(f"  Retrieval: {self.top_k_retrieval} docs")
        if self.use_reranker:
            print(f"  Reranked: {self.top_k_reranked} docs")
        print(f"  Max Context: {self.max_context_length:,} chars")
        
        # Logging
        print(f"\n{BLUE}Logging:{RESET}")
        print(f"  Debug: {'✅' if self.enable_debug else '❌'}")
        print(f"  Performance: {'✅' if self.log_performance else '❌'}")
        print(f"  Save Results: {'✅' if self.save_results else '❌'}")
        
        # Models
        print(f"\n{BLUE}Models:{RESET}")
        print(f"  LLM: {self.llm_model}")
        print(f"  Embeddings: {self.embedding_model.split('/')[-1]}")
        if self.use_reranker:
            print(f"  Reranker: {self.reranker_model.split('/')[-1]}")


# ANSI escape codes for colors
BLUE = "\033[96m"
GREEN = "\033[92m"  
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


def debug_log(debug_data: dict[str, any], config: RagConfig):
    """Enhanced debug logging with configurable verbosity."""
    if not config.enable_debug:
        return
    
    print("\n\n" + f"{GREEN}" + "=" * 175)
    print("DEBUG INFORMATION")
    print("=" * 175 + f"{RESET}")

    for key, value in debug_data.items():
        if key == "retrieved_docs" or key == "reranked_docs":
            print(f"\n\n{GREEN}Retrieved Documents Overview{RESET}")
            print(f"{GREEN}" + "=" * 175 + f"{RESET}")

            print(f"\n{BLUE}Distribución de documentos por capítulo{RESET}")
            print(f"{BLUE}" + "-" * 175 + f"{RESET}")
            chapter_dist: dict[str, int] = {}
            for doc in value:
                chapter = doc.metadata.get("chapter_title", "Sin título")
                if chapter not in chapter_dist:
                    chapter_dist[chapter] = 1
                else:
                    chapter_dist[chapter] += 1

            pprint(chapter_dist)

            for i, doc in enumerate(value):
                print(f"\n\n{BLUE}Document #{i}{RESET}")
                print(f"{BLUE}" + "-" * 175 + f"{RESET}")
                print("Metadata:")
                pprint(doc.metadata)

                print("\nContent:")
                content = doc.page_content.strip()
                preview = content[:1000] + ("..." if len(content) > 1000 else "")
                print(preview)

        else:
            print(f"\n\n{GREEN}{key.capitalize()}{RESET}")
            print(f"{GREEN}" + "=" * 175 + f"{RESET}")
            if isinstance(value, (list, dict)):
                pprint(value)
            else:
                print(value)
