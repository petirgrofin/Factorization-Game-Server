import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.generator import run_generate_polynomials, run_generate_questions
from server.agent_generator import run_agent
from server.validator import validate_polynomial

import re

def strip_formatting(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("$", "")
    expr = re.sub(r"\\[a-zA-Z]+", "", expr)
    expr = re.sub(r"[{}\[\]]", "", expr)
    return expr.strip()

app = FastAPI()

class PlayerData(BaseModel):
    accuracy: float
    avg_time: float
    recent_mistakes: list[str]

# --- Generator.py endpoints ---

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/generate-polynomials")
def generate_polynomials(data: PlayerData):
    player_data = {
        "accuracy": data.accuracy,
        "avg_time": data.avg_time,
        "recent_mistakes": data.recent_mistakes
    }

    result = run_generate_polynomials(player_data)
    raw_exercises = [strip_formatting(p) for p in result["raw"]]

    validated_exercises = []
    for exercise in raw_exercises:
        if validate_polynomial(exercise):
            validated_exercises.append(exercise)
        else:
            validated_exercises.append("")

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
        raw = result["raw"].strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        questions = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Model returned invalid JSON")

    # print(questions)

    return {
        "questions": questions,
        "difficulty": result["difficulty"],
        "topic": result["topic"]
    }


# --- Agent endpoints ---

@app.post("/agent/generate-polynomials")
def agent_generate_polynomials(data: PlayerData):
    player_data = {
        "accuracy": data.accuracy,
        "avg_time": data.avg_time,
        "recent_mistakes": data.recent_mistakes
    }

    try:
        result = run_agent({**player_data, "force_mode": "polynomial"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    if result["mode"] != "polynomial":
        raise HTTPException(status_code=500, detail="Agent returned wrong mode")

    raw = result["raw"]

    # raw may be a list already (parsed JSON) or a string
    if isinstance(raw, str):
        try:
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Agent returned invalid JSON for polynomials")

    validated_exercises = []
    for exercise in raw:
        cleaned = strip_formatting(str(exercise))
        validated_exercises.append(cleaned if validate_polynomial(cleaned) else "")

    return {
        "polynomials": validated_exercises,
        "difficulty": result["difficulty"],
        "topic": result["topic"],
        "mode": result["mode"]
    }


@app.post("/agent/generate-questions")
def agent_generate_questions(data: PlayerData):
    player_data = {
        "accuracy": data.accuracy,
        "avg_time": data.avg_time,
        "recent_mistakes": data.recent_mistakes
    }

    try:
        result = run_agent({**player_data, "force_mode": "question"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    if result["mode"] != "question":
        raise HTTPException(status_code=500, detail="Agent returned wrong mode")

    raw = result["raw"]

    if isinstance(raw, str):
        try:
            raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Agent returned invalid JSON for questions")

    return {
        "questions": raw,
        "difficulty": result["difficulty"],
        "topic": result["topic"],
        "mode": result["mode"]
    }