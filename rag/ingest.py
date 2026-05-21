import json
from pathlib import Path

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DATA_PATH = Path(__file__).parent / "data" / "factorizations.json"
CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"


with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

for item in data:
    text = f"Topic: {item['topic']}\nDifficulty: {item['difficulty']}\nExpression: {item['expression']}"
    documents.append(
        Document(
            page_content=text,
            metadata={
                "topic": item["topic"],
                "difficulty": item["difficulty"],
                "expression": item["expression"]
            }
        )
    )

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    task_type="retrieval_document"
)

texts = [doc.page_content for doc in documents]
metadatas = [doc.metadata for doc in documents]

# Embed one by one to avoid batching bug
print(f"Embedding {len(texts)} documents one by one...")
vectors = []
for i, text in enumerate(texts):
    vec = embeddings.embed_documents([text])  # list of one
    vectors.append(vec[0])
    print(f"  [{i+1}/{len(texts)}] done")

print(f"Got {len(vectors)} embeddings")

db = Chroma(
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH)
)

db._collection.add(
    ids=[str(i) for i in range(len(texts))],
    embeddings=vectors,
    documents=texts,
    metadatas=metadatas
)

print(f"Vector database created successfully with {len(texts)} documents.")