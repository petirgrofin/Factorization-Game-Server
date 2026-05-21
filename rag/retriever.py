from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path

CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2",
    task_type="retrieval_query"
)

db = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embeddings
)

def retrieve_examples(topic: str, difficulty: str, k: int = 5):

    docs = db.similarity_search(
        query=f"{topic} {difficulty}",
        k=k,
        filter={
            "$and": [
                {"topic": {"$eq": topic}},
                {"difficulty": {"$eq": difficulty}}
            ]
        }
    )

    return [
        doc.metadata["expression"]
        for doc in docs
    ]