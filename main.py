from fastapi import FastAPI
from pydantic import BaseModel

from generator import generate_exercise
from validator import validate_exercise

app = FastAPI()

class PlayerData(BaseModel):
    accuracy: float
    avg_time: float
    recent_mistakes: list[str]

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/generate")
def generate(data: PlayerData):

    player_data = {
        "accuracy": data.accuracy,
        "avg_time": data.avg_time,
        "recent_mistakes": data.recent_mistakes
    }

    result = generate_exercise(player_data)

    raw_exercises = [
        p.strip()
        for p in result["raw"].split(',')
    ]

    validated_exercises = []

    for exercise in raw_exercises:
        if validate_exercise(exercise):
            validated_exercises.append(exercise)
        else:
            validated_exercises.append("")

    return {
        "polynomials": validated_exercises
    }