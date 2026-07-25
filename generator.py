from collections import Counter
import json

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from server.prompts import load_prompt
from server.rag.retriever import retrieve_examples
from server.formatter import format_mistakes_for_retrieval, format_mistakes_for_prompt

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
)


VALID_TOPICS = {
    "common_factor",
    "perfect_square_trinomial",
    "difference_of_squares",
    "inspection",
}


def determine_focus_topic(mistakes: list[dict]) -> str:
    polynomial_topics = [
        mistake["method"]
        for mistake in mistakes
        if (
            mistake.get("type") == "polynomial"
            and mistake.get("method") in VALID_TOPICS
        )
    ]

    if polynomial_topics:
        return Counter(polynomial_topics).most_common(1)[0][0]

    # Fallback when there are no polynomial mistakes.
    return "common_factor"


def determine_difficulty(
    accuracy: float,
    avg_time: float,
) -> str:
    if accuracy < 0.5:
        difficulty = "easy"
    elif accuracy < 0.75:
        difficulty = "medium"
    elif accuracy < 0.9:
        difficulty = "hard"
    else:
        difficulty = "very_hard"

    return difficulty


def analyze_player(data: dict) -> tuple[str, str]:
    difficulty = determine_difficulty(
        accuracy=data["accuracy"],
        avg_time=data["avg_time"],
    )

    topic = determine_focus_topic(
        data.get("mistakes", [])
    )

    return difficulty, topic

def extract_response_text(response) -> str:
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif (
                isinstance(item, dict)
                and "text" in item
            ):
                text_parts.append(
                    str(item["text"])
                )

        return "".join(text_parts)

    raise ValueError(
        "Unsupported model response format"
    )

def parse_json_response(raw_text: str):
    cleaned_text = raw_text.strip()

    # Some models may wrap JSON in a Markdown code fence.
    if cleaned_text.startswith("```"):
        lines = cleaned_text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        cleaned_text = "\n".join(lines).strip()

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "The model did not return valid JSON. "
            f"Received:\n{raw_text}"
        ) from exc

polynomial_chain = load_prompt("polynomial") | llm
question_chain = load_prompt("question") | llm

def run_generate_polynomials(player_data: dict):
    mistakes = player_data.get("mistakes", [])

    difficulty, topic = analyze_player(
        player_data
    )

    retrieval_mistakes = (
        format_mistakes_for_retrieval(mistakes)
    )

    examples = retrieve_examples(
        topic=topic,
        difficulty=difficulty,
        mistakes=retrieval_mistakes,
        k=5,
    )

    mistakes_description = (
        format_mistakes_for_prompt(mistakes)
    )

    print("Mistakes sent to model:")
    print(mistakes_description)

    print("\nRetrieved examples:")
    print(examples)

    response = polynomial_chain.invoke(
        {
            "accuracy": player_data["accuracy"],
            "avg_time": player_data["avg_time"],
            "topic": topic,
            "difficulty": difficulty,
            "mistakes": mistakes_description,
            "examples": "\n\n".join(examples),
        }
    )

    raw_text = extract_response_text(response)
    parsed = parse_json_response(raw_text)

    return {
        "raw": parsed,
        "difficulty": difficulty,
        "topic": topic,
    }


def run_generate_questions(player_data: dict):
    mistakes = player_data.get("mistakes", [])

    difficulty, topic = analyze_player(
        player_data
    )

    mistakes_description = (
        format_mistakes_for_prompt(mistakes)
    )

    response = question_chain.invoke(
        {
            "accuracy": player_data["accuracy"],
            "avg_time": player_data["avg_time"],
            "topic": topic,
            "difficulty": difficulty,
            "mistakes": mistakes_description,
        }
    )

    return {
        "raw": extract_response_text(response),
        "difficulty": difficulty,
        "topic": topic,
    }