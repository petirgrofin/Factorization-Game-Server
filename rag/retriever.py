from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pathlib import Path

CHROMA_PATH = Path(__file__).parent.parent / "chroma_db"

from dotenv import load_dotenv

load_dotenv()

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
    mistakes: str,
    k: int = 5
):
    query = f"""
    {topic}
    {difficulty}
    {mistakes}
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