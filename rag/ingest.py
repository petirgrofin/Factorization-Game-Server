import json
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from features import polynomial_description

load_dotenv()

DATA_PATH = Path(__file__).parent / "data" / "factorizations.json"
CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

for item in data:

    expression = (
        item["expression"]
        .replace("–", "-")
        .replace("−", "-")
    )

    description = polynomial_description(expression)

    text = f"""
    Topic: {item['topic']}
    Difficulty: {item['difficulty']}

    Expression:
    {expression}

    Description:
    {description}
    """

    documents.append(
        Document(
            page_content=text.strip(),
            metadata={
                "topic": item["topic"],
                "difficulty": item["difficulty"],
                "expression": expression
            }
        )
    )

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    task_type="retrieval_document"
)

texts = [doc.page_content for doc in documents]
metadatas = [doc.metadata for doc in documents]

print(f"Embedding {len(texts)} documents one by one...")

vectors = []

for i, text in enumerate(texts):

    vec = embeddings.embed_documents([text])

    if not vec or len(vec) == 0:
        print(f"Failed embedding at index {i}")
        continue

    vectors.append(vec[0])

    print(f"[{i+1}/{len(texts)}] done")

    db = Chroma(
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH)
)

db._collection.add(
    ids=[str(i) for i in range(len(vectors))],
    embeddings=vectors,
    documents=texts[:len(vectors)],
    metadatas=metadatas[:len(vectors)]
)

print(f"Stored {len(documents)} documents.")