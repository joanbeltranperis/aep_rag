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
    config.basic_setup()

    pipeline = RAGPipeline(config)

    documents, vector_store = load_documents_and_vector_store(config)

    question = "Buenos días, tengo una bebe que nació en 2017 que va a salir de la comunidad autónoma antes de cumplir los 4 meses durante 2 ó 3. Mi duda es si vacunando a los 2 meses, podríamos adelantar la vacunación de los 4: HEXA+MEN C+PNEUMO a los 3 y medio, o dicho de otra manera cual es el intervalo minimo que tiene q pasar entre la 1ª y 2ª vacunación HEXA y PNEUMO. Saludos y gracias por vuestra inestimable labor"

    human_answer = "El intervalo mínimo entre dosis de la VNC13 es de 4 semanas; entre las dosis de hexavalente en una pauta 2+1 es de 2 meses. Neis-Vac permite una sola dosis si se administra a partir de los 4 meses, si es antes precisa de 2 dosis separadas por 2 meses.\nAmplie información en nuestras tablas del Manual de Vacunas\nUn cordial saludo"

    pipeline.process(
        question=question,
        documents=documents,
        vector_store=vector_store,
        human_answer=human_answer.strip(),
    )


if __name__ == "__main__":
    main()
