import copy
import os

import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from termcolor import colored

from config import RagConfig


def create_vector_stores(config: RagConfig) -> tuple[FAISS, FAISS]:
    """
    Create two vector stores:
    1. Original documents (full sections)
    2. Split documents (chunked into smaller pieces)

    Returns:
        tuple[FAISS, FAISS]: (original_store, split_store)
    """
    print(colored("Starting document ingestion process...", "blue"))

    print(colored("Downloading and parsing documents...", "blue"))
    documents = []
    successful_chapters = 0

    for chapter_number in range(1, config.total_chapters + 1):
        print(f"Processing chapter {chapter_number}/{config.total_chapters}...")

        parsed_documents = parse_document_from_url(
            config.base_url.format(chapter_number=chapter_number),
        )

        if parsed_documents:
            documents.extend(parsed_documents)
            successful_chapters += 1
        else:
            print(f"Failed to parse chapter {chapter_number}")

    print(
        colored(
            f"Successfully parsed {successful_chapters}/{config.total_chapters} chapters",
            "green",
        )
    )
    print(colored(f"Total documents extracted: {len(documents)}", "green"))

    if not documents:
        raise ValueError("No documents were successfully parsed!")

    print(colored("Creating original document vector store...", "blue"))
    original_store = FAISS.from_documents(documents, config.embedding_model_instance)

    os.makedirs(config.vector_store_path, exist_ok=True)
    original_store.save_local(config.vector_store_path)
    print(
        colored(f"Original vector store saved to: {config.vector_store_path}", "green")
    )

    print(colored("Creating split document vector store...", "blue"))
    print(f"Chunk size: {config.chunk_size} characters")
    print(f"Chunk overlap: {config.chunk_overlap} characters")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=config.length_function,
        separators=config.separators,
        is_separator_regex=False,
    )

    split_documents = []
    total_chunks = 0

    for i, doc in enumerate(documents):
        if i % 10 == 0:
            print(f"Splitting document {i + 1}/{len(documents)}...")

        chunks = text_splitter.split_documents([doc])

        for chunk in chunks:
            chunk.metadata = doc.metadata.copy()

        split_documents.extend(chunks)
        total_chunks += len(chunks)

    print(
        colored(f"Split {len(documents)} documents into {total_chunks} chunks", "green")
    )

    split_store = FAISS.from_documents(split_documents, config.embedding_model_instance)

    os.makedirs(config.split_vector_store_path, exist_ok=True)
    split_store.save_local(config.split_vector_store_path)
    print(
        colored(
            f"Split vector store saved to: {config.split_vector_store_path}", "green"
        )
    )

    print(colored("Vector Store Creation Summary:", "blue"))
    print(f"Original documents: {len(documents)}")
    print(f"Split chunks: {total_chunks}")
    print(f"Original store path: {config.vector_store_path}")
    print(f"Split store path: {config.split_vector_store_path}")

    return original_store, split_store


def pretty_print_document(doc: Document, doc_id: int) -> str:
    """Format a Document object in academic-style string format."""

    meta_lines = [f"'{k}': '{v}'" for k, v in doc.metadata.items()]
    metadata_str = "{\n" + "\n".join("    " + line for line in meta_lines) + "\n}"

    content_preview = doc.page_content.strip()

    return (
        f"Document #{doc_id}\nMetadata:\n{metadata_str}\nContent:\n{content_preview}\n"
    )


def parse_document_from_url(url: str) -> list[Document]:
    """Parse a document from the given URL and return a list of langchain documents."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        soup_full = BeautifulSoup(response.content, "html.parser")  # contains tables
        soup_text_only = copy.deepcopy(soup_full)  # will have tables removed

        # Remove tables from the version used to extract text content
        for table in soup_text_only.find_all("table"):
            table.decompose()

        chapter_title_element = soup_text_only.find("h1")
        chapter_title = (
            chapter_title_element.get_text(strip=True)
            if chapter_title_element
            else "Unknown Chapter"
        )

        chapter_number = url.split("cap-")[-1] if "cap-" in url else "unknown"

        documents = []
        sections = soup_text_only.find_all(["h3", "h4"])

        for section in sections:
            heading_text = section.get_text(strip=True)
            if not heading_text:
                continue

            content_parts = []
            current = section.next_sibling

            while current:
                if current.name in ["h3", "h4"]:
                    break

                if hasattr(current, "get_text"):
                    text = current.get_text(strip=True)
                    if text:
                        content_parts.append(text)
                elif isinstance(current, str):
                    text = current.strip()
                    if text:
                        content_parts.append(text)

                current = current.next_sibling

            if content_parts:
                content = "\n".join(content_parts)

                metadata = {
                    "url": url,
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                    "heading": heading_text,
                }

                if section.name == "h4":
                    for prev_elem in section.find_all_previous(["h3"]):
                        parent_heading = prev_elem.get_text(strip=True)
                        if parent_heading:
                            metadata["heading"] = parent_heading
                            metadata["subheading"] = heading_text
                            break

                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)

        table_documents = parse_tables(url, soup_full)

        for table_doc in table_documents:
            table_doc.metadata.update(
                {
                    "url": url,
                    "chapter_number": chapter_number,
                    "chapter_title": chapter_title,
                }
            )

        return documents + table_documents

    except requests.RequestException as e:
        print(colored(f"Request error for {url}: {str(e)}", "red"))
        return []
    except Exception as e:
        print(colored(f"Parse error for {url}: {str(e)}", "red"))
        return []


def parse_tables(url: str, soup: BeautifulSoup) -> list[Document]:
    """Parse and serialize HTML tables with titles inferred from <strong> tags."""
    try:
        html_tables = pd.read_html(url)
        tables = []

        html_table_tags = soup.find_all("table")

        for table_index, (df, table_tag) in enumerate(
            zip(html_tables, html_table_tags)
        ):
            table_title = "Tabla sin título"
            previous_strongs = table_tag.find_all_previous("strong", limit=3)
            if previous_strongs:
                table_title = previous_strongs[0].get_text(strip=True)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [" - ".join(map(str, col)).strip() for col in df.columns]

            rows = df.to_dict(orient="records")

            for row_index, row in enumerate(rows):
                text = "\n".join(
                    f"{str(col).strip()}: {str(val).strip() if val is not None else ''}"
                    for col, val in row.items()
                )

                tables.append(
                    Document(
                        page_content=f"Tabla {table_index + 1}, fila {row_index + 1}:\n{text}",
                        metadata={
                            "url": url,
                            "table_index": table_index,
                            "row_index": row_index,
                            "content_type": "table_row",
                            "table_title": table_title,
                        },
                    )
                )

        return tables

    except Exception as e:
        print(colored(f"Error parsing tables from {url}: {str(e)}", "yellow"))
        return []


if __name__ == "__main__":
    """Run this script to create/recreate the vector stores."""
    from config import RagConfig

    print(colored("Running vector store creation script...", "blue"))

    config = RagConfig()

    if not hasattr(config, "base_url"):
        print(colored("Error: base_url not configured in RagConfig", "red"))
        print(colored("Please add base_url to your config.py file", "yellow"))
        exit(1)

    try:
        original_store, split_store = create_vector_stores(config)
        print(colored("Vector stores created successfully!", "green"))
        print(colored("You can now run your main application.", "blue"))

    except Exception as e:
        print(colored(f"Error creating vector stores: {str(e)}", "red"))
        raise
