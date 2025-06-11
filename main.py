import json
import os

from langchain_community.vectorstores import FAISS

from config import RagConfig, GREEN, BLUE, YELLOW, RESET
from pipeline import RAGPipeline
from vector_stores import create_vector_stores


def load_questions(path: str) -> dict[str, any]:
    """Load questions from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_questions(data: dict[str, any], path: str):
    """Save questions to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def run_single_question_demo(config: RagConfig, documents: list, vector_store):
    """Demonstrate the pipeline with a single question."""
    pipeline = RAGPipeline(config)
    
    question = "Buenos días, con los cambios del calendario vacunal nos surgen algunas dudas en el centro. He visitado un niño en la consulta que se siguió la pauta de vacunación con el calendario nuevo. Ahora tiene 6 meses: a los 2 meses recibió hexavalente + antineumocócica; a los 4 m hexavalente + antineumocócica pero en lugar de recibir una NeisVac, lleva puesta una MCC. Mi duda es si hay que poner una NeisVac de refuerzo para completar la pauta y en qué momento habría que ponerla. Muchas gracias!!"
    human_answer = "Según el capítulo 28. Hepatitis A, …"
    
    print(f"\n{GREEN}🔄 Processing single question demo{RESET}")
    print(f"{BLUE}Question: {question[:100]}...{RESET}")
    
    result = pipeline.process_question(
        question=question,
        documents=documents,
        vector_store=vector_store,
        human_answer=human_answer if config.enable_evaluation else None
    )
    
    print(f"\n{GREEN}✅ Demo completed!{RESET}")
    print(f"Answer length: {len(result['answer'])} characters")
    
    if result['evaluation']:
        print(f"Evaluation: {result['evaluation']}")
    
    return result


def run_batch_evaluation(config: RagConfig, documents: list, vector_store, start_idx: int = 0, end_idx: int = 10):
    """Run batch evaluation on multiple questions."""
    if not config.enable_evaluation:
        print(f"{YELLOW}⚠️  Evaluation is disabled in current configuration{RESET}")
        return
    
    pipeline = RAGPipeline(config)
    
    try:
        json_questions = load_questions("questions_evaluation.json")
    except FileNotFoundError:
        print(f"{YELLOW}⚠️  questions_evaluation.json not found, skipping batch evaluation{RESET}")
        return
    
    questions_list = json_questions.get("questions", [])
    
    print(f"\n{GREEN}🔄 Running batch evaluation ({start_idx} to {end_idx}){RESET}")
    
    for i in range(start_idx, min(end_idx, len(questions_list))):
        try:
            json_question = questions_list[i]
            question = json_question["question"]
            human_answer = json_question.get("human_answer", {}).get("content", "")
            
            print(f"\n{BLUE}Processing question {i + 1}/{len(questions_list)}{RESET}")
            
            result = pipeline.process_question(
                question=question,
                documents=documents,
                vector_store=vector_store,
                human_answer=human_answer
            )
            
            # Update the question data
            json_question["rag_answer"] = {
                "content": result["answer"],
                "evaluation": result["evaluation"].get("rag_evaluation", {}),
                "metrics": {
                    "total_time": result["metrics"].total_time,
                    "retrieval_time": result["metrics"].retrieval_time,
                    "generation_time": result["metrics"].generation_time,
                    "documents_retrieved": result["metrics"].documents_retrieved,
                    "context_length": result["metrics"].context_length
                }
            }
            
            if "human_answer" not in json_question:
                json_question["human_answer"] = {"content": human_answer}
            
            json_question["human_answer"]["evaluation"] = result["evaluation"].get("human_evaluation", {})
            
            # Save progress
            if config.save_results:
                save_questions(json_questions, "questions_evaluation.json")
                print(f"{GREEN}✅ Saved progress for question {i + 1}{RESET}")
        
        except Exception as e:
            print(f"{YELLOW}❌ Error processing question {i + 1}: {e}{RESET}")
            if "questions" in json_questions and i < len(json_questions["questions"]):
                json_questions["questions"][i]["error"] = str(e)
                if config.save_results:
                    save_questions(json_questions, "questions_evaluation.json")
            continue
    
    print(f"\n{GREEN}✅ Batch evaluation completed!{RESET}")


def main():
    """Main function with configurable pipeline execution."""
    print(f"{GREEN}🚀 RAG System Starting{RESET}")
    
    # Create configuration
    config = RagConfig()
    
    # =================================================================
    # 🎛️  EASY CONFIGURATION - Choose your setup
    # =================================================================
    
    # Choose one setup (uncomment the one you want):
    # config.enable_all()         # Enable all features (reranker, evaluation, debug, etc.)
    # config.basic_setup()        # Basic setup (no reranker, no evaluation, faster)
    
    # Or manually configure what you need:
    config.use_reranker = True           # Enable/disable reranking
    config.enable_evaluation = False     # Enable/disable evaluation
    config.enable_debug = False          # Enable/disable debug logging
    config.log_performance = True       # Enable/disable performance metrics
    
    # Adjust retrieval if needed:
    # config.set_retrieval_size(30, 15)   # Retrieve 30, rerank to 15
    
    # =================================================================
    
    # Print current configuration
    config.print_status()
    
    # Load or create document store
    print(f"\n{BLUE}📚 Loading document store...{RESET}")
    if not os.path.exists("document_store"):
        print(f"{BLUE}Creating new document store...{RESET}")
        document_store = create_vector_stores(config)
    else:
        print(f"{BLUE}Loading existing document store...{RESET}")
        document_store = FAISS.load_local(
            "document_store",
            embeddings=config.embedding_model_instance,
            allow_dangerous_deserialization=True,
        )
    
    documents = list(document_store.docstore._dict.values())
    print(f"{GREEN}✅ Loaded {len(documents)} documents{RESET}")
    
    # Run demonstrations based on configuration
    if config.enable_evaluation:
        # Run single question demo with evaluation
        run_single_question_demo(config, documents, document_store)
        
        # Optionally run batch evaluation
        user_choice = input(f"\n{BLUE}Run batch evaluation? (y/n): {RESET}").lower().strip()
        if user_choice == 'y':
            start_idx = int(input(f"{BLUE}Start index (default 0): {RESET}") or "0")
            end_idx = int(input(f"{BLUE}End index (default 10): {RESET}") or "10")
            run_batch_evaluation(config, documents, document_store, start_idx, end_idx)
    else:
        # Run single question demo without evaluation
        run_single_question_demo(config, documents, document_store)
    
    print(f"\n{GREEN}🎉 RAG System Demo Completed!{RESET}")


# =====================================================================
# 💡 USAGE EXAMPLES - Simple configuration examples
# =====================================================================

def example_default():
    """Example: Default configuration."""
    config = RagConfig()
    # Uses default settings: reranker enabled, evaluation disabled
    return config

def example_full_features():
    """Example: All features enabled."""
    config = RagConfig()
    config.enable_all()  # Enable everything
    return config

def example_basic():
    """Example: Basic minimal setup."""
    config = RagConfig()
    config.basic_setup()  # Minimal components for speed
    return config

def example_custom():
    """Example: Custom configuration."""
    config = RagConfig()
    
    # Pick exactly what you want
    config.use_reranker = False        # Disable for speed
    config.enable_evaluation = True    # But enable evaluation
    config.enable_debug = True        # And debugging
    
    return config


if __name__ == "__main__":
    main()
