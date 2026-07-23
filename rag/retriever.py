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

def retrieve_examples(
    topic: str,
    difficulty: str,
    mistakes: list[str],
    k: int = 5
):

    query = f"""
    Factorization exercises.

    Topic:
    {topic}

    Difficulty:
    {difficulty}

    Student weaknesses:
    {", ".join(mistakes)}

    Retrieve structurally similar exercises.
    Prefer exercises matching the same factorization patterns.
    """

    docs = db.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=20,
        filter={
            "$and": [
                {"topic": {"$eq": topic}},
                {"difficulty": {"$eq": difficulty}}
            ]
        }
    )

    return [
        doc.page_content
        for doc in docs
    ]