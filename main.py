import json
import re

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Annotated, Literal, Union

from server.generator import run_generate_polynomials, run_generate_questions
from server.agent_generator import run_agent
from server.validator import validate_polynomial
from server.mistake_interpreter import enrich_mistakes


def strip_formatting(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("$", "")
    expr = re.sub(r"\\[a-zA-Z]+", "", expr)
    expr = re.sub(r"[{}\[\]]", "", expr)
    return expr.strip()


class PolynomialMistake(BaseModel):
    type: Literal["polynomial"] = "polynomial"

    original_polynomial: str
    current_factorization: str

    method_id: str | None = None
    method: Literal[
        "common_factor",
        "perfect_square_trinomial",
        "difference_of_squares",
        "inspection"
    ]

    target_factor: str
    inputs: list[str]


class QuestionMistake(BaseModel):
    type: Literal["question"] = "question"
    question_text: str
    player_response: bool
    correct_response: bool | None = None


Mistake = Annotated[
    Union[PolynomialMistake, QuestionMistake],
    Field(discriminator="type")
]


class PlayerData(BaseModel):
    accuracy: float
    avg_time: float
    mistakes: list[Mistake] = Field(default_factory=list)


def serialize_player_data(data: PlayerData) -> dict:
    player_data = data.model_dump()

    player_data["mistakes"] = enrich_mistakes(
        player_data["mistakes"]
    )

    return player_data


app = FastAPI()


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/generate-polynomials")
def generate_polynomials(data: PlayerData):
    serialized_data = serialize_player_data(data)
    print(serialized_data)
    result = run_generate_polynomials(serialized_data)

    raw_exercises = [
        strip_formatting(str(polynomial))
        for polynomial in result["raw"]
    ]

    validated_exercises = [
        exercise if validate_polynomial(exercise) else ""
        for exercise in raw_exercises
    ]

    return {
        "polynomials": validated_exercises,
        "difficulty": result["difficulty"],
        "topic": result["topic"]
    }


@app.post("/generate-questions")
def generate_questions(data: PlayerData):
    result = run_generate_questions(serialize_player_data(data))

    raw = result["raw"]

    if isinstance(raw, str):
        try:
            cleaned = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            raw = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Model returned invalid JSON"
            ) from exc

    return {
        "questions": raw,
        "difficulty": result["difficulty"],
        "topic": result["topic"]
    }


@app.post("/agent/generate-polynomials")
def agent_generate_polynomials(data: PlayerData):
    player_data = serialize_player_data(data)

    try:
        result = run_agent({
            **player_data,
            "force_mode": "polynomial"
        })
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {exc}"
        ) from exc

    if result["mode"] != "polynomial":
        raise HTTPException(
            status_code=500,
            detail="Agent returned wrong mode"
        )

    raw = result["raw"]

    if isinstance(raw, str):
        try:
            cleaned = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            raw = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Agent returned invalid JSON for polynomials"
            ) from exc

    if not isinstance(raw, list):
        raise HTTPException(
            status_code=500,
            detail="Agent polynomial output must be a list"
        )

    validated_exercises = []

    for exercise in raw:
        cleaned = strip_formatting(str(exercise))
        validated_exercises.append(
            cleaned if validate_polynomial(cleaned) else ""
        )

    return {
        "polynomials": validated_exercises,
        "difficulty": result["difficulty"],
        "topic": result["topic"],
        "mode": result["mode"]
    }


@app.post("/agent/generate-questions")
def agent_generate_questions(data: PlayerData):
    player_data = serialize_player_data(data)

    try:
        result = run_agent({
            **player_data,
            "force_mode": "question"
        })
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {exc}"
        ) from exc

    if result["mode"] != "question":
        raise HTTPException(
            status_code=500,
            detail="Agent returned wrong mode"
        )

    raw = result["raw"]

    if isinstance(raw, str):
        try:
            cleaned = (
                raw.strip()
                .removeprefix("```json")
                .removeprefix("```")
                .removesuffix("```")
                .strip()
            )
            raw = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=500,
                detail="Agent returned invalid JSON for questions"
            ) from exc

    return {
        "questions": raw,
        "difficulty": result["difficulty"],
        "topic": result["topic"],
        "mode": result["mode"]
    }