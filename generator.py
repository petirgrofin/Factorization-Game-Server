import json
from typing import Counter

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from server.prompts import load_prompt
from server.rag.retriever import retrieve_examples
from server.formatter import format_mistakes_for_retrieval, format_mistakes_for_prompt

from server.difficulty import (
    use_computed_difficulties,
)
from server.exercise_plan import (
    create_exercise_plan,
)
from server.utils import (
    extract_response_text,
    get_plan_distribution,
    normalize_hint_balloons,
    parse_json_response,
)

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


def build_retrieval_examples(
    exercise_plan: list[dict],
    mistakes: list[dict],
) -> str:
    retrieval_mistakes = format_mistakes_for_retrieval(mistakes)

    sections: list[str] = []

    for item in exercise_plan:
        examples = retrieve_examples(
            topic=item["method"],
            difficulty=item["difficulty"],
            mistakes=retrieval_mistakes,
            k=max(2, item["count"]),
        )

        section = "\n".join(
            [
                f"Method: {item['method']}",
                ("Difficulty: " f"{item['difficulty']}"),
                ("Required count: " f"{item['count']}"),
                ("Category: " f"{item['category']}"),
                "Examples:",
                *examples,
            ]
        )

        sections.append(section)

    return "\n\n".join(sections)


polynomial_chain = load_prompt("polynomial") | llm
question_chain = load_prompt("question") | llm
hint_chain = load_prompt("hint") | llm


def run_generate_polynomials(
    player_data: dict,
):
    mistakes = player_data.get(
        "mistakes",
        [],
    )

    mistakes_description = format_mistakes_for_prompt(mistakes)

    method_analysis = use_computed_difficulties(player_data["methods"])

    exercise_plan = create_exercise_plan(method_analysis)

    examples = build_retrieval_examples(
        exercise_plan=exercise_plan,
        mistakes=mistakes,
    )

    print("Mistakes sent to model:")
    print(mistakes_description)

    print("\nMethod analysis:")
    print(
        json.dumps(
            method_analysis,
            indent=2,
        )
    )

    print("\nExercise plan:")
    print(
        json.dumps(
            exercise_plan,
            indent=2,
        )
    )

    print("\nRetrieved examples:")
    print(examples)

    response = polynomial_chain.invoke(
        {
            "method_performance": json.dumps(
                method_analysis,
                indent=2,
            ),
            "exercise_plan": json.dumps(
                exercise_plan,
                indent=2,
            ),
            "examples": examples,
            "total_exercises": 6,
        }
    )

    print(f"Response:")
    print(response)

    raw_text = extract_response_text(response)

    parsed = parse_json_response(raw_text)

    return {
        "raw": parsed,
    }


VALID_DIFFICULTIES = {
    "easy",
    "medium",
    "hard",
    "very_hard",
}


def validate_question_response(
    parsed: object,
    exercise_plan: list[dict],
    expected_count: int,
) -> list[dict]:
    if not isinstance(parsed, list):
        raise ValueError(
            "The question model response must be a JSON array."
        )

    if len(parsed) != expected_count:
        raise ValueError(
            "The question model returned "
            f"{len(parsed)} questions; expected {expected_count}."
        )

    normalized_questions: list[dict] = []

    for expected_id, question in enumerate(
        parsed,
        start=1,
    ):
        if not isinstance(question, dict):
            raise ValueError(
                f"Question {expected_id} must be a JSON object."
            )

        method = question.get("method")
        difficulty = question.get("difficulty")
        answer = question.get("respuesta_correcta")

        if not isinstance(answer, bool):
            raise ValueError(
                f"Question {expected_id} must have a boolean "
                "as respuesta_correcta."
            )

        if method not in VALID_TOPICS:
            raise ValueError(
                f"Question {expected_id} has an invalid method: "
                f"{method!r}."
            )

        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"Question {expected_id} has an invalid difficulty: "
                f"{difficulty!r}."
            )

        if not isinstance(answer, bool):
            raise ValueError(
                f"Question {expected_id} must have a boolean "
                "as respuesta_correcta."
            )

        contexto = str(
            question.get("contexto", "")
        ).strip()

        pregunta = str(
            question.get("pregunta", "")
        ).strip()

        explicacion = str(
            question.get("explicacion", "")
        ).strip()

        if not contexto:
            raise ValueError(
                f"Question {expected_id} has no contexto."
            )

        if not pregunta:
            raise ValueError(
                f"Question {expected_id} has no pregunta."
            )

        if not pregunta.endswith("?"):
            raise ValueError(
                f"Question {expected_id} must end with '?'."
            )

        if not explicacion:
            raise ValueError(
                f"Question {expected_id} has no explicacion."
            )

        normalized_questions.append(
            {
                "id": expected_id,
                "method": method,
                "difficulty": difficulty,
                "contexto": contexto,
                "pregunta": pregunta,
                "respuesta_correcta": answer,
                "explicacion": explicacion,
            }
        )

    expected_distribution = get_plan_distribution(
        exercise_plan,
    )

    actual_distribution = Counter(
        (
            question["method"],
            question["difficulty"],
        )
        for question in normalized_questions
    )

    if actual_distribution != expected_distribution:
        raise ValueError(
            "The generated questions do not follow the "
            "exercise plan.\n"
            f"Expected: {dict(expected_distribution)}\n"
            f"Received: {dict(actual_distribution)}"
        )

    # answers = Counter(
       # question["respuesta_correcta"]
       # for question in normalized_questions
    # )

    # if answers[True] != 5 or answers[False] != 5:
        # raise ValueError(
           #  "The generated questions must contain exactly "
           #  "5 true and 5 false answers. "
           #  f"Received: {dict(answers)}"
        # )

    return normalized_questions


def run_generate_questions(
    player_data: dict,
):
    method_analysis = use_computed_difficulties(
        player_data["methods"],
    )

    exercise_plan = create_exercise_plan(
        method_analysis,
        total_exercises=10,
    )

    print("\nQuestion method analysis:")
    print(
        json.dumps(
            method_analysis,
            indent=2,
        )
    )

    print("\nQuestion exercise plan:")
    print(
        json.dumps(
            exercise_plan,
            indent=2,
        )
    )

    response = question_chain.invoke(
        {
            "method_performance": json.dumps(
                method_analysis,
                indent=2,
                ensure_ascii=False,
            ),
            "exercise_plan": json.dumps(
                exercise_plan,
                indent=2,
                ensure_ascii=False,
            ),
            "total_questions": 10,
        }
    )

    print("\nQuestion model response:")
    print(response)

    raw_text = extract_response_text(response)
    parsed = parse_json_response(raw_text)

    validated_questions = validate_question_response(
        parsed=parsed,
        exercise_plan=exercise_plan,
        expected_count=10,
    )

    return {
        "raw": validated_questions,
    }


def validate_hint_response(
    parsed: object,
    hint_number: int,
) -> dict:
    if not isinstance(parsed, dict):
        raise ValueError("The model hint response must be " "a JSON object.")

    balloons = normalize_hint_balloons(parsed.get("balloons"))

    methods = parsed.get(
        "methods",
        [],
    )

    if not isinstance(methods, list):
        methods = []

    methods = [method for method in methods if method in VALID_TOPICS]

    return {
        "hint_number": hint_number,
        "balloons": balloons,
        "methods": methods,
        "is_solution": hint_number == 3,
    }


def run_generate_hint(
    hint_data: dict,
):
    polynomial = hint_data["polynomial"]
    current_factorization = hint_data.get(
        "current_factorization",
        "",
    )

    mistakes = hint_data.get(
        "mistakes",
        [],
    )

    hints_asked = hint_data.get(
        "hints_asked",
        0,
    )

    hint_number = hints_asked + 1

    mistakes_description = format_mistakes_for_prompt(mistakes)

    if not mistakes_description.strip():
        mistakes_description = (
            "El estudiante todavía no ha cometido "
            "errores registrados en este ejercicio."
        )

    print("\nGenerating hint:")
    print(f"Polynomial: {polynomial}")
    print("Current factorization: " f"{current_factorization}")
    print(f"Hint number: {hint_number}")

    print("\nMistakes sent to hint model:")
    print(mistakes_description)

    response = hint_chain.invoke(
        {
            "polynomial": polynomial,
            "current_factorization": (
                current_factorization or "Todavía no hay una factorización."
            ),
            "hint_number": hint_number,
            "mistakes": mistakes_description,
            "available_methods": json.dumps(
                sorted(VALID_TOPICS),
                ensure_ascii=False,
            ),
        }
    )

    print("\nHint model response:")
    print(response)

    raw_text = extract_response_text(response)

    parsed = parse_json_response(raw_text)

    validated = validate_hint_response(
        parsed=parsed,
        hint_number=hint_number,
    )

    return {
        "raw": validated,
    }
