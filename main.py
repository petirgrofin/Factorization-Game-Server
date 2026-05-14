import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from generator import run_generate_polynomials, run_generate_questions
from validator import validate_polynomial

import re

def strip_formatting(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("$", "")
    expr = re.sub(r"\\[a-zA-Z]+", "", expr)  # remove LaTeX commands e.g. \cdot, \frac
    expr = re.sub(r"[{}\[\]]", "", expr)      # remove brackets
    return expr.strip()

app = FastAPI()

class PlayerData(BaseModel):
    accuracy: float
    avg_time: float
    recent_mistakes: list[str]

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/generate_polynomials")
def generate_polynomials(data: PlayerData):
    player_data = {
        "accuracy": data.accuracy,
        "avg_time": data.avg_time,
        "recent_mistakes": data.recent_mistakes
    }

    result = run_generate_polynomials(player_data)

    raw_exercises = [strip_formatting(p) for p in result["raw"].split(',')]

    print(raw_exercises)

    # Unlike the questions, polynomials can be validated directly with SymPy.
    validated_exercises = []
    for exercise in raw_exercises:
        if validate_polynomial(exercise):
            validated_exercises.append(exercise)
        else:
            validated_exercises.append("") # if the game detects an empty string it will
            # default to an existing exercise

    return {"polynomials": validated_exercises}


@app.post("/generate-questions")
def generate_questions(data: PlayerData):
    player_data = {
        "accuracy": data.accuracy,
        "avg_time": data.avg_time,
        "recent_mistakes": data.recent_mistakes
    }

    result = run_generate_questions(player_data)

    try:
        # Strip markdown fences if the model wraps output in ```json ... ```
        raw = result["raw"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        questions = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON")

    return {
        "questions": questions,        # list of {question, answer}
        "difficulty": result["difficulty"],
        "topic": result["topic"]
    }