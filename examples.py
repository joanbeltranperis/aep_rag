"""
Simple examples showing how to use the RAG system with different configurations.
"""

import os
from langchain_community.vectorstores import FAISS

from config import RagConfig, GREEN, BLUE, RESET
from pipeline import RAGPipeline
from vector_stores import create_vector_stores


def example_1_default():
    """Default configuration - good balance of features."""
    print(f"\n{GREEN}📝 Example 1: Default Configuration{RESET}")
    
    config = RagConfig()
    # Default: chapter filtering + reranker enabled, evaluation disabled
    config.print_status()
    
    return config


def example_2_all_features():
    """Enable all features for maximum functionality."""
    print(f"\n{GREEN}📝 Example 2: All Features Enabled{RESET}")
    
    config = RagConfig()
    config.enable_all()  # Enable everything
    
    return config


def example_3_basic_setup():
    """Basic setup for speed - minimal components."""
    print(f"\n{GREEN}📝 Example 3: Basic Setup{RESET}")
    
    config = RagConfig()
    config.basic_setup()  # Minimal for speed
    
    return config


def example_4_custom_configuration():
    """Custom configuration - pick exactly what you need."""
    print(f"\n{GREEN}📝 Example 4: Custom Configuration{RESET}")
    
    config = RagConfig()
    
    # Configure exactly what you want
    config.use_reranker = False         # Disable for speed
    config.enable_evaluation = True     # But enable evaluation
    config.enable_debug = True         # And debugging
    config.log_performance = True     # Track performance
    
    # Adjust retrieval size
    config.set_retrieval_size(25, 12) # Smaller for testing
    
    config.print_status()
    
    return config


def example_5_toggle_features():
    """Example of toggling features at runtime."""
    print(f"\n{GREEN}📝 Example 5: Toggle Features{RESET}")
    
    config = RagConfig()
    
    print(f"{BLUE}Initial state:{RESET}")
    config.print_status()
    
    # Toggle some features
    print(f"\n{BLUE}Toggling features...{RESET}")
    config.toggle_reranker(False)      # Disable reranker
    config.toggle_evaluation(True)     # Enable evaluation
    config.toggle_debug(True)         # Enable debug
    
    print(f"\n{BLUE}After toggling:{RESET}")
    config.print_status()
    
    return config


def example_6_complete_workflow():
    """Complete workflow example with actual question processing."""
    print(f"\n{GREEN}📝 Example 6: Complete Workflow{RESET}")
    
    # Setup configuration
    config = RagConfig()
    config.use_reranker = True
    config.enable_debug = False        # Keep output clean
    config.log_performance = True     # But track performance
    
    # Load documents (if available)
    if not os.path.exists("document_store"):
        print(f"{BLUE}Document store not found. Run main.py first to create it.{RESET}")
        return config
    
    document_store = FAISS.load_local(
        "document_store",
        embeddings=config.embedding_model_instance,
        allow_dangerous_deserialization=True,
    )
    
    documents = list(document_store.docstore._dict.values())
    print(f"{GREEN}✅ Loaded {len(documents)} documents{RESET}")
    
    # Create pipeline
    pipeline = RAGPipeline(config)
    
    # Process a question
    question = "¿Cuál es la pauta de vacunación para hepatitis B en recién nacidos?"
    
    print(f"\n{BLUE}Processing: {question}{RESET}")
    
    result = pipeline.process_question(
        question=question,
        documents=documents,
        vector_store=document_store
    )
    
    print(f"\n{GREEN}Results:{RESET}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Documents used: {result['metrics'].documents_retrieved} → {result['metrics'].documents_reranked}")
    print(f"Time: {result['metrics'].total_time:.2f}s")
    
    return result


def compare_configurations():
    """Compare the two main setup options side by side."""
    print(f"\n{GREEN}📊 Configuration Comparison{RESET}")
    
    configs = {
        "Default": RagConfig(),
        "All Features": RagConfig(),
        "Basic": RagConfig()
    }
    
    # Configure each
    configs["All Features"].enable_all()
    configs["Basic"].basic_setup()
    
    # Print comparison table
    print(f"\n{BLUE}Setup Comparison:{RESET}")
    print(f"{'Component':<20} {'Default':<10} {'All Features':<12} {'Basic':<8}")
    print("-" * 55)
    
    components = [
        ("Chapter Filtering", "use_chapter_filtering"),
        ("Reranker", "use_reranker"),
        ("Evaluation", "enable_evaluation"),
        ("Debug Logging", "enable_debug"),
        ("Performance Log", "log_performance"),
        ("Save Results", "save_results"),
    ]
    
    for name, attr in components:
        row = f"{name:<20}"
        for config in configs.values():
            value = "✅" if getattr(config, attr) else "❌"
            row += f" {value:<10}"
        print(row)
    
    # Print retrieval sizes
    print(f"\n{BLUE}Retrieval Sizes:{RESET}")
    print(f"{'Setup':<12} {'Initial':<8} {'Reranked':<8}")
    print("-" * 30)
    for name, config in configs.items():
        reranked = config.top_k_reranked if config.use_reranker else "N/A"
        print(f"{name:<12} {config.top_k_retrieval:<8} {reranked:<8}")


def run_all_examples():
    """Run all examples to demonstrate different configurations."""
    print(f"{GREEN}🚀 Running RAG configuration examples{RESET}")
    
    examples = [
        example_1_default,
        example_2_all_features,
        example_3_basic_setup,
        example_4_custom_configuration,
        example_5_toggle_features,
        example_6_complete_workflow,
    ]
    
    results: dict[str, any] = {}
    
    for example in examples:
        try:
            print(f"\n" + "="*60)
            result = example()
            results[example.__name__] = result
        except Exception as e:
            print(f"{GREEN}❌ Error in {example.__name__}: {e}{RESET}")
            results[example.__name__] = None
    
    # Run comparison
    print(f"\n" + "="*60)
    compare_configurations()
    
    print(f"\n{GREEN}🎉 All examples completed!{RESET}")
    return results


if __name__ == "__main__":
    run_all_examples() 