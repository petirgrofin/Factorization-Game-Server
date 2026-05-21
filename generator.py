from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from prompts import load_prompt
from rag.retriever import retrieve_examples
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.7
)

def analyze_player(data):
    accuracy = data["accuracy"]
    mistakes = data["recent_mistakes"]

    focus_topic = max(set(mistakes), key=mistakes.count)

    if accuracy < 0.5:
        difficulty = "easy"
    elif accuracy < 0.75:
        difficulty = "normal"
    elif accuracy < 0.9:
        difficulty = "hard"
    else:
        difficulty = "very_hard"

    return difficulty, focus_topic

polynomial_chain = load_prompt("polynomial") | llm
question_chain = load_prompt("question") | llm

def run_generate_polynomials(player_data):
    difficulty, topic = analyze_player(player_data)

    examples = retrieve_examples(
        topic=topic,
        difficulty=difficulty,
        k=5
    )

    print("Examples retrieved: ", examples)

    response = polynomial_chain.invoke({
        "accuracy": player_data["accuracy"],
        "avg_time": player_data["avg_time"],
        "topic": topic,
        "difficulty": difficulty,
        "examples": "\n".join(examples)
    })

    parsed = json.loads(response.content)

    return {
        "raw": parsed,
        "difficulty": difficulty,
        "topic": topic
    }


def run_generate_questions(player_data):
    difficulty, topic = analyze_player(player_data)

    response = question_chain.invoke({
        "accuracy": player_data["accuracy"],
        "avg_time": player_data["avg_time"],
        "topic": topic,
        "difficulty": difficulty
    })

    return {
        "raw": response.content,
        "difficulty": difficulty,
        "topic": topic
    }