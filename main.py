import os

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import ChatPromptTemplate
from templates.titles import document_titles
from templates.prompts import initial_retrieval_prompt, final_answer_prompt

from config import RagConfig, debug_log
from vector_stores import create_vector_stores


os.environ["GOOGLE_API_KEY"] = "AIzaSyCIfd-QrhcRqrsTE1CosgAftdt_6Kskvm8"


def main():
    config = RagConfig()

    query = "¿Qué hacer si no se dispone de documentación sobre las vacunas administradas previamente a un niño inmigrante?"

    template = ChatPromptTemplate.from_template(
        initial_retrieval_prompt,
    )

    prompt = template.format_messages(
        document_titles=document_titles,
        user_question=query,
    )

    response = config.generative_model.invoke(prompt)
    relevant_documents = response.content.strip().split(", ")

    if not os.path.exists("document_store"):
        document_store = create_vector_stores(config)

    else:
        document_store = FAISS.load_local(
            "document_store",
            embeddings=config.embedding_model,
            allow_dangerous_deserialization=True,
        )

    docs = document_store.docstore._dict.values()
    documents = []
    for doc in docs:
        if doc.metadata["chapter_number"] in relevant_documents:
            documents.append(doc)

    temp_faiss = FAISS.from_documents(
        documents,
        config.embedding_model,
    )

    retrieved_docs = temp_faiss.similarity_search(
        query,
        k=config.top_k,
    )

    template = ChatPromptTemplate.from_template(final_answer_prompt)
    context = ""
    for doc in retrieved_docs:
        meta = doc.metadata
        context += f"{meta}\n{doc.page_content}\n"

    prompt = template.format_messages(
        question=query,
        context=context,
    )
    response = config.generative_model.invoke(prompt)
    print(response.content)

    debug_data = {
        "query": query,
        "relevant_documents": relevant_documents,
        "retrieved_docs": retrieved_docs,
    }
    debug_log(debug_data)


if __name__ == "__main__":
    main()
