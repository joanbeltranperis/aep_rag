import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import RagConfig


def create_vector_stores(config: RagConfig) -> FAISS:
    documents = []
    for chapter_number in range(1, config.total_chapters + 1):
        parsed_documents = parse_document_from_url(
            config.base_url.format(chapter_number=chapter_number),
        )

        documents.extend(parsed_documents)

    document_store = FAISS.from_documents(documents, config.embedding_model)
    document_store.save_local("document_store")

    return document_store


def parse_document_from_url(url: str) -> list[Document]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        content = soup.find_all("div", class_="field-item odd")[1]

    except Exception:
        print(f"Error: unable to parse content from {url}")
        return []

    if content is None:
        print(f"Error: content is None for {url}")
        return []

    for a in content.find_all("a"):
        if a.get("title") == "Índice":
            a.decompose()

    # tables = parse_tables(url)

    chapter_title = (
        soup.title.string.split("|")[0].strip()
        if soup.title and soup.title.string
        else ""
    )

    special_spans = content.find_all(
        "span",
        style=lambda s: s
        and "font-weight: bold" in s
        and "text-transform: uppercase" in s,
    )

    for span in special_spans:
        span.name = "h4"

        span_parent = span.parent
        span_parent.insert_before(span)

        if not span_parent.get_text(strip=True):
            span_parent.decompose()

    text_documents = []
    heading_str = None
    for heading in content.find_all(["h3", "h4"]):
        subheading_str = None

        if heading.name == "h3":
            heading_str = heading.get_text(strip=True)
            if (
                "bibliografía" in heading_str.lower()
                or "enlaces de interés" in heading_str.lower()
                or "historial de actualizaciones" in heading_str.lower()
            ):
                continue
        elif heading.name == "h4":
            subheading_str = heading.get_text(strip=True)

        section_text = []
        for sibling in heading.find_next_siblings():
            if sibling.name in ["h3", "h4"]:
                break

            text = sibling.get_text(strip=True)
            section_text.append(text)

        content_str = "\n".join(section_text)

        document = Document(
            page_content=content_str,
            metadata={
                "chapter_number": chapter_title.split(".")[0],
                "url": url,
                "chapter_title": chapter_title,
            },
        )

        if heading_str:
            document.metadata["heading"] = heading_str
        if subheading_str:
            document.metadata["subheading"] = subheading_str

        if heading.name == "h3":
            subheading_str = None

        text_documents.append(document)

    documents = text_documents

    return documents


def parse_tables(url: str) -> list[Document]:
    tables = []
    html_tables = pd.read_html(url)

    for i, table in enumerate(html_tables):
        df = html_tables[i]
        row_list = df.to_dict(orient="records")

        for row in row_list:
            tables.append(
                Document(
                    page_content=str(row),
                    metadata={
                        "url": url,
                    },
                )
            )

    return tables
