import json

from fastapi import FastAPI, HTTPException

from server.generator import (
    run_generate_polynomials,
    run_generate_questions,
    run_generate_hint,
)
from server.difficulty import (
    analyze_methods,
    get_method_actions,
    get_new_difficulties,
)
from server.validator import validate_polynomial
from server.models import DifficultyRequest, GenerationRequest, HintRequest, HintResponse
from server.utils import (
    normalize_expression_text,
    serialize_generation_data,
    serialize_hint_data,
    strip_formatting,
)


app = FastAPI()


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/compute-difficulties")
def compute_difficulties(data: DifficultyRequest):
    method_analysis = analyze_methods(data.methods.model_dump())

    methods = {
        method: {
            "difficulty": result["new_difficulty"],
            "action": result["action"],
        }
        for method, result in method_analysis.items()
    }

    return {
        "methods": methods,
        "new_difficulties": get_new_difficulties(method_analysis),
        "method_actions": get_method_actions(method_analysis),
        "method_analysis": method_analysis,
    }


@app.post("/generate-polynomials")
def generate_polynomials(
    data: GenerationRequest,
):
    serialized_data = serialize_generation_data(data)

    result = run_generate_polynomials(serialized_data)

    print("Parsed model response:")
    print(
        json.dumps(
            result["raw"],
            indent=2,
        )
    )

    validated_exercises = []

    for item in result["raw"]:
        if not isinstance(item, dict):
            continue

        polynomial = strip_formatting(
            str(
                item.get(
                    "polynomial",
                    "",
                )
            )
        )

        validated_exercises.append(
            {
                "polynomial": polynomial,
                "method": item.get("method"),
                "difficulty": item.get("difficulty"),
            }
        )

    return {
        "polynomials": validated_exercises,
    }


@app.post("/generate-questions")
def generate_questions(data: GenerationRequest):
    try:
        result = run_generate_questions(
            serialize_generation_data(data)
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return {
        "questions": result["raw"],
    }

@app.post(
    "/generate-hint",
    response_model=HintResponse,
)
def generate_hint(
    data: HintRequest,
):
    polynomial = strip_formatting(data.polynomial)

    current_factorization = strip_formatting(data.current_factorization)

    if not validate_polynomial(polynomial):
        raise HTTPException(
            status_code=422,
            detail=("The original polynomial " "is not valid."),
        )

    normalized_polynomial = normalize_expression_text(polynomial)

    for mistake in data.mistakes:
        normalized_mistake_polynomial = normalize_expression_text(
            mistake.original_polynomial
        )

        if normalized_mistake_polynomial != normalized_polynomial:
            raise HTTPException(
                status_code=422,
                detail=("All mistakes must belong " "to the requested polynomial."),
            )

    serialized_data = serialize_hint_data(data)

    serialized_data["polynomial"] = polynomial

    serialized_data["current_factorization"] = current_factorization

    try:
        result = run_generate_hint(serialized_data)
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        print("Error while generating hint:")
        print(exc)

        raise HTTPException(
            status_code=502,
            detail=("The hint could not be " "generated."),
        ) from exc

    return result["raw"]
