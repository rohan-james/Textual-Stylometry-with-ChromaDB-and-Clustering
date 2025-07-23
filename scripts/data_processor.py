import os
import requests
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

RAW_TEXTS_DIR = "../data/raw_texts"
CHROMA_DB_PATH = "../chroma_db_data"
COLLECTION_NAME = "literary_styles"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BOOKS_TO_PROCESS = [
    ("Pride and Prejudice", "Jane Austen", 1342),
    ("The Call of Cthulhu", "H.P. Lovecraft", 55227),
    ("The Adventures of Tom Sawyer", "Mark Twain", 74),
    ("Frankenstein", "Mary Shelley", 84),
]


def download_gutenberg_text(book_id, title, author, output_dir):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    filepath = os.path.join(
        output_dir, f"{title.replace('','_')}_{author.replace('','_')}.txt"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text_content = response.text
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text_content)
        return text_content
    except requests.exceptions.RequestException as e:
        return None


def clean_text(text):
    if not text:
        return ""

    start_marker = "*** Start of Book"
    end_marker = "*** End of Book"

    start_index = text.find(start_marker)
    if start_index != -1:
        text = text[text.find("***", start_index + len(start_marker)) + 3 :].strip()

    end_index = text.find(end_marker)
    if end_index != -1:
        text = text[:end_index].strip()

    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[\t]+", " ", text)
    text = text.strip()
    return text


def segment_text_into_paragraph(text, min_len=50):
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if len(p.strip()) > min_len]


def main():
    os.makedirs(RAW_TEXTS_DIR, exist_ok=True)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    print(f"Loading the current sentence transformer model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=sentence_transformer_ef
    )

    documents_to_add = []
    metadatas_to_add = []
    ids_to_add = []
    embeddings_to_add = []

    total_paragraphs = 0

    for title, author, book_id in BOOKS_TO_PROCESS:
        raw_text = download_gutenberg_text(book_id, title, author, RAW_TEXTS_DIR)
        if raw_text:
            cleaned_text = clean_text(raw_text)
            paragraphs = segment_text_into_paragraph(cleaned_text)
            total_paragraphs += len(paragraphs)

            batch_size = 32
            for i in range(0, len(paragraphs), batch_size):
                batch_paragraphs = paragraphs[i : i + batch_size]
                batch_embeddings = embedding_model.encode(
                    batch_paragraphs, convert_to_tensor=False
                ).tolist()

                for j, paragraph in enumerate(batch_paragraphs):
                    paragraph_id = f"{book_id}-{i+j}"
                    documents_to_add.append(paragraph)
                    embeddings_to_add.append(batch_embeddings[j])
                    metadatas_to_add.append(
                        {
                            "book_title": title,
                            "author": author,
                            "gutenberg_id": book_id,
                            "paragraph_id_in_book": i + j,
                        }
                    )
                    ids_to_add.append(paragraph_id)

    if documents_to_add:
        print(f"Adding {len(documents_to_add)} documents to ChromaDB")
        collection.add(
            documents=documents_to_add,
            embeddings=embeddings_to_add,
            metadatas=metadatas_to_add,
            ids=ids_to_add,
        )
        print("data ingestion complete")
    else:
        print("no documents to add")

    # Testing a simple query
    results = collection.query(
        query_texts=["a man walking through a dark, mysterious house"], n_results=2
    )

    for i, doc in enumerate(results["documents"][0]):
        print(f"result {i+1}\n")
        print(f"Text: \n {doc[:100]}")
        print(f"Author: {results['metadatas'][0][i]['author']}")
        print(f"Book: {results['metadatas'][0][i]['book_title']}")


if __name__ == "__main__":
    main()
