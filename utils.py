from collections import Counter
import json
import re

from server.mistake_interpreter import enrich_mistakes
from server.models import GenerationRequest, HintRequest


def to_sympy_format(expr_str: str) -> str:
    # Replace ^ with **
    expr_str = expr_str.replace("^", "**")

    # Insert * between numbers and variables
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)

    # Insert * between variables and parentheses
    expr_str = re.sub(r'([a-zA-Z])\(', r'\1*(', expr_str)

    # Insert * between closing/opening parentheses
    expr_str = re.sub(r'\)\(', r')*(', expr_str)

    return expr_str


def strip_formatting(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("$", "")
    expr = re.sub(r"\\[a-zA-Z]+", "", expr)
    expr = re.sub(r"[{}\[\]]", "", expr)
    return expr.strip()


def normalize_expression_text(expression: str) -> str:
    return re.sub(r"\s+", "", strip_formatting(expression))


def serialize_generation_data(data: GenerationRequest) -> dict:
    generation_data = data.model_dump()
    generation_data["mistakes"] = enrich_mistakes(generation_data["mistakes"])
    return generation_data


def serialize_hint_data(data: HintRequest) -> dict:
    hint_data = data.model_dump()
    hint_data["mistakes"] = enrich_mistakes(hint_data["mistakes"])
    return hint_data


def extract_response_text(response) -> str:
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                text_parts.append(str(item["text"]))

        return "".join(text_parts)

    raise ValueError("Unsupported model response format")


def parse_json_response(raw_text: str):
    cleaned_text = raw_text.strip()

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


def get_plan_distribution(exercise_plan: list[dict]) -> Counter:
    return Counter(
        (item["method"], item["difficulty"])
        for item in exercise_plan
        for _ in range(item["count"])
    )


def normalize_hint_balloons(balloons) -> list[str]:
    if isinstance(balloons, str):
        balloons = [balloons]

    if not isinstance(balloons, list):
        raise ValueError("The model response must contain a list named 'balloons'.")

    normalized: list[str] = []

    for balloon in balloons:
        text = str(balloon).strip()

        if not text:
            continue

        if len(text) > 280:
            text = text[:277].rstrip() + "..."

        normalized.append(text)

    if not normalized:
        raise ValueError("The model returned no usable hint text.")

    return normalized[:2]
