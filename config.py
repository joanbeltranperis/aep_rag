from typing import Any
import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from pprint import pprint

os.environ["GOOGLE_API_KEY"] = "AIzaSyCIfd-QrhcRqrsTE1CosgAftdt_6Kskvm8"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


class RagConfig:
    def __init__(self):
        self.total_chapters = 53
        self.base_url = r"https://vacunasaep.org/documentos/manual/cap-{chapter_number}"
        self.generative_model = genai.GenerativeModel("gemini-2.0-flash-lite")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"
        )
        self.top_k = 20


# ANSI escape codes for colors
BLUE = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"


def debug_log(debug_data: dict[str, Any]):
    print("\n\n" + f"{GREEN}" + "=" * 175)
    print("DEBUG INFORMATION")
    print("=" * 175 + f"{RESET}")

    for key, value in debug_data.items():
        if key == "retrieved_docs":
            print(f"\n\n{GREEN}Retrieved Documents Overview{RESET}")
            print(f"{GREEN}" + "=" * 175 + f"{RESET}")

            print(f"\n{BLUE}Distribución de documentos por capítulo{RESET}")
            print(f"{BLUE}" + "-" * 175 + f"{RESET}")
            chapter_dist = {}
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
                pprint(doc.metadata, width=120)

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
