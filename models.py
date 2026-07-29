from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


Difficulty = Literal[
    "easy",
    "medium",
    "hard",
    "very_hard",
]

MethodName = Literal[
    "common_factor",
    "perfect_square_trinomial",
    "difference_of_squares",
    "inspection",
]

DifficultyAction = Literal["promoted", "stayed", "relegated"]


class PolynomialMistake(BaseModel):
    type: Literal["polynomial"] = "polynomial"
    original_polynomial: str
    current_factorization: str
    method_id: str | None = None
    method: MethodName
    target_factor: str
    inputs: list[str]


class QuestionMistake(BaseModel):
    type: Literal["question"] = "question"
    question_text: str
    player_response: bool
    correct_response: bool | None = None


Mistake = Annotated[
    Union[PolynomialMistake, QuestionMistake],
    Field(discriminator="type"),
]


class MethodPerformance(BaseModel):
    accuracy: float = Field(ge=0.0, le=1.0)
    difficulty: Difficulty


class MethodsPerformance(BaseModel):
    common_factor: MethodPerformance
    perfect_square_trinomial: MethodPerformance
    difference_of_squares: MethodPerformance
    inspection: MethodPerformance


class PlayerData(BaseModel):
    methods: MethodsPerformance
    avg_time: float | None = None
    mistakes: list[Mistake] = Field(default_factory=list)


class MethodDifficulty(BaseModel):
    difficulty: Difficulty
    action: DifficultyAction


class MethodsDifficulty(BaseModel):
    common_factor: MethodDifficulty
    perfect_square_trinomial: MethodDifficulty
    difference_of_squares: MethodDifficulty
    inspection: MethodDifficulty


class DifficultyRequest(BaseModel):
    methods: MethodsPerformance


class GenerationRequest(BaseModel):
    methods: MethodsDifficulty
    avg_time: float | None = None
    mistakes: list[Mistake] = Field(default_factory=list)


class HintRequest(BaseModel):
    polynomial: str = Field(min_length=1)
    current_factorization: str = ""
    mistakes: list[PolynomialMistake] = Field(default_factory=list)
    hints_asked: int = Field(
        ge=0,
        le=2,
        description="Number of hints already displayed for this polynomial.",
    )


class HintResponse(BaseModel):
    hint_number: Literal[1, 2, 3]
    balloons: list[str] = Field(min_length=1, max_length=2)
    methods: list[MethodName] = Field(default_factory=list)
    is_solution: bool
