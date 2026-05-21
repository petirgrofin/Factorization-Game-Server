from pathlib import Path

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"

print("Using database path:")
print(CHROMA_PATH)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    task_type="retrieval_query"
)

db = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings
)

print("\nCollection count:")
print(db._collection.count())

print("\nGetting first few raw documents...")

peek = db._collection.peek(limit=5)

print("\nDocuments:")
for doc in peek["documents"]:
    print(doc)

print("\nMetadata:")
for meta in peek["metadatas"]:
    print(meta)

print("\nRunning similarity search...\n")

docs = db.similarity_search(
    query="difference_of_squares hard",
    k=5
)

print(f"Retrieved {len(docs)} documents\n")

for i, doc in enumerate(docs, start=1):
    print(f"--- Result {i} ---")
    print(doc.page_content)
    print(doc.metadata)
    print()