#!/usr/bin/env python3
"""
RAG Q&A System for Immunization Manual
Main execution script with configurable architecture components.
"""

from config import RagConfig
from pipeline import RAGPipeline
from utils import load_documents_and_vector_store


def main():
    config = RagConfig()
    config.enable_all()
    config.debug_mode = True
    config.log_stats = True
    config.enable_evaluation = True
    config.use_text_splitter = False
    config.use_chapter_filtering = True


    pipeline = RAGPipeline(config)

    documents, vector_store = load_documents_and_vector_store(config)

    question = "Paciente de 3 años que acude por herpes zoster. Como antecedente destaca varicela dos semanas antes de cumplir 6 meses. ¿Sería conveniente vacunar de la varicela (dado que la infección fue antes de los 6 meses)? O al pasar el herpes zóster se entiende que está inmunizado. En caso que haya que vacunarle, cuántas dosis y cuánto tiempo tras zóster. Gracias"

    human_answer = "Si el diagnóstico de Herpes Zoster es certero, en estos casos NO se necesita proseguir con la vacunación, pues se entiende que son individuos no susceptibles tras este evento. Le recomendamos la lectura del siguiente capítulo del Manual de Vacunas.\nUn saludo"

    pipeline.process(
        question=question,
        documents=documents,
        vector_store=vector_store,
        human_answer=human_answer.strip(),
    )


if __name__ == "__main__":
    main()
