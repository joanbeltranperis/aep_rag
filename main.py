#!/usr/bin/env python3
"""
RAG Q&A System for Immunization Manual
Main execution script with configurable architecture components.
"""

from config import RagConfig
from pipeline import RAGPipeline
from utils import load_documents


def main():
    """Main execution function for the RAG Q&A system."""

    config = RagConfig()
    config.enable_all()
    config.debug_mode = True
    config.enable_evaluation = True

    documents, vector_store = load_documents(config)
    pipeline = RAGPipeline(config)

    question = "Buenos días, ¿Qué periodo de tiempo se maneja para que un caso de parotiditis pueda ser debido a la vacuna, en este caso tetravírica? Muchas gracias."

    human_answer = "En el caso de un niño vacunado de lactante (12 meses) para confirmar que la parotiditis es infecciosa debe hacer PCR en faringe o en orina porque los anticuerpos no tienen validez en el diagnóstico al estar vacunado.\nLas vacunas SRP o varicela tardan 15 días aproximadamente en hacer su efecto.\nUn cordial saludo"

    print("\n🚀 Starting RAG Q&A System")
    print(f"📊 Configuration: {config.get_active_components()}")

    pipeline.process(
        question=question,
        documents=documents,
        vector_store=vector_store,
        human_answer=human_answer,
    )

    print("\n🎉 RAG System Processing Completed!")


if __name__ == "__main__":
    main()
