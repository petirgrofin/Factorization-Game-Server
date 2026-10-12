import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Query

from server import database

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
from server.models import (
    DifficultyRequest, GenerationRequest, HintRequest, HintResponse,
    PlayerRegistration, SessionEnd, SessionStart, TelemetryEvent,
)
from server.utils import (
    normalize_expression_text,
    serialize_generation_data,
    serialize_hint_data,
    strip_formatting,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    database.initialize_database()
    yield


app = FastAPI(lifespan=lifespan)


def _identity(
    x_player_id: str | None,
    x_session_id: str | None,
    *,
    required: bool = False,
) -> tuple[str | None, str | None]:
    if required and not x_player_id:
        raise HTTPException(status_code=401, detail="X-Player-ID header is required.")
    if x_player_id:
        if len(x_player_id) > 128 or not x_player_id.strip():
            raise HTTPException(status_code=400, detail="Invalid player id.")
        try:
            database.ensure_player(x_player_id)
            if x_session_id:
                database.start_session(x_player_id, x_session_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    elif x_session_id:
        raise HTTPException(status_code=400, detail="A session id requires a player id.")
    return x_player_id, x_session_id


def _log(player_id, session_id, event_type, payload, **fields):
    if player_id:
        database.record_event(
            player_id, event_type, session_id=session_id, payload=payload, **fields
        )


def _log_mistakes(player_id, session_id, endpoint, mistakes):
    for mistake in mistakes:
        _log(
            player_id, session_id, "mistake_received", mistake,
            endpoint=endpoint, method=mistake.get("method"),
            polynomial=mistake.get("original_polynomial"),
            question_text=mistake.get("question_text"),
            player_input=mistake.get("inputs", mistake.get("player_response")),
            correct=False,
        )


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/compute-difficulties")
def compute_difficulties(
    data: DifficultyRequest,
    x_player_id: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    player_id, session_id = _identity(x_player_id, x_session_id)
    method_analysis = analyze_methods(data.methods.model_dump())

    methods = {
        method: {
            "difficulty": result["new_difficulty"],
            "action": result["action"],
        }
        for method, result in method_analysis.items()
    }

    response = {
        "methods": methods,
        "new_difficulties": get_new_difficulties(method_analysis),
        "method_actions": get_method_actions(method_analysis),
        "method_analysis": method_analysis,
    }
    for method, result in method_analysis.items():
        _log(
            player_id, session_id, "difficulty_changed", result,
            direction="server_to_client", endpoint="/compute-difficulties",
            method=method, difficulty=result["new_difficulty"],
            previous_difficulty=result["previous_difficulty"], action=result["action"],
        )
    return response


@app.post("/generate-polynomials")
def generate_polynomials(
    data: GenerationRequest,
    x_player_id: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    player_id, session_id = _identity(x_player_id, x_session_id)
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

    response = {
        "polynomials": validated_exercises,
    }
    for exercise in validated_exercises:
        _log(
            player_id, session_id, "polynomial_sent", exercise,
            direction="server_to_client", endpoint="/generate-polynomials",
            polynomial=exercise["polynomial"], method=exercise.get("method"),
            difficulty=exercise.get("difficulty"),
        )
    _log_mistakes(
        player_id, session_id, "/generate-polynomials",
        serialized_data.get("mistakes", []),
    )
    return response


@app.post("/generate-questions")
def generate_questions(
    data: GenerationRequest,
    x_player_id: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    player_id, session_id = _identity(x_player_id, x_session_id)
    serialized_data = serialize_generation_data(data)
    try:
        result = run_generate_questions(
            serialized_data
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    response = {
        "questions": result["raw"],
    }
    for question in result["raw"]:
        _log(
            player_id, session_id, "question_sent", question,
            direction="server_to_client", endpoint="/generate-questions",
            method=question.get("method"), difficulty=question.get("difficulty"),
            question_text=question.get("pregunta", question.get("question_text")),
        )
    _log_mistakes(
        player_id, session_id, "/generate-questions",
        serialized_data.get("mistakes", []),
    )
    return response

@app.post(
    "/generate-hint",
    response_model=HintResponse,
)
def generate_hint(
    data: HintRequest,
    x_player_id: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    player_id, session_id = _identity(x_player_id, x_session_id)
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

    _log(
        player_id, session_id, "hint_requested", serialized_data,
        endpoint="/generate-hint", polynomial=polynomial,
        player_input=current_factorization,
    )
    _log(
        player_id, session_id, "hint_sent", result["raw"],
        direction="server_to_client", endpoint="/generate-hint",
        polynomial=polynomial,
    )
    return result["raw"]


@app.put("/players/{player_id}")
def register_player(player_id: str, data: PlayerRegistration):
    if not player_id.strip() or len(player_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid player id.")
    database.ensure_player(player_id, data.metadata)
    return {"player_id": player_id}


@app.post("/sessions", status_code=201)
def create_session(
    data: SessionStart,
    x_player_id: str | None = Header(default=None),
):
    player_id, _ = _identity(x_player_id, None, required=True)
    try:
        session_id = database.start_session(player_id, data.session_id, data.metadata)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    database.record_event(
        player_id, "session_started", session_id=session_id,
        payload={"metadata": data.metadata},
    )
    return {"player_id": player_id, "session_id": session_id}


@app.patch("/sessions/{session_id}/end")
def close_session(
    session_id: str,
    data: SessionEnd,
    x_player_id: str | None = Header(default=None),
):
    player_id, _ = _identity(x_player_id, None, required=True)
    if not database.end_session(player_id, session_id, data.duration_seconds):
        raise HTTPException(status_code=404, detail="Open session not found.")
    database.record_event(
        player_id, "session_ended", session_id=session_id,
        elapsed_seconds=data.duration_seconds,
    )
    return {"status": "ended"}


@app.post("/telemetry/events", status_code=201)
def create_telemetry_event(
    data: TelemetryEvent,
    x_player_id: str | None = Header(default=None),
    x_session_id: str | None = Header(default=None),
):
    player_id, session_id = _identity(
        x_player_id, x_session_id, required=True
    )
    event_id = database.record_event(
        player_id, data.event_type, session_id=session_id,
        direction=data.direction,
        occurred_at=data.occurred_at.isoformat() if data.occurred_at else None,
        method=data.method, difficulty=data.difficulty,
        previous_difficulty=data.previous_difficulty, action=data.action,
        polynomial=data.polynomial, question_text=data.question_text,
        player_input=data.player_input, correct=data.correct,
        elapsed_seconds=data.elapsed_seconds, payload=data.payload,
    )
    return {"event_id": event_id}


@app.get("/players/{player_id}/timeline")
def player_timeline(
    player_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    x_player_id: str | None = Header(default=None),
):
    authenticated, _ = _identity(x_player_id, None, required=True)
    if authenticated != player_id:
        raise HTTPException(status_code=403, detail="Cannot access another player.")
    return {"player_id": player_id, "events": database.get_timeline(player_id, limit, offset)}


@app.get("/players/{player_id}/metrics")
def player_metrics(
    player_id: str,
    x_player_id: str | None = Header(default=None),
):
    authenticated, _ = _identity(x_player_id, None, required=True)
    if authenticated != player_id:
        raise HTTPException(status_code=403, detail="Cannot access another player.")
    summary = database.get_player_summary(player_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Player not found.")
    return summary
