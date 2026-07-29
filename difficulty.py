from typing import Literal, TypeAlias

Difficulty: TypeAlias = Literal[
    "easy",
    "medium",
    "hard",
    "very_hard",
]

MethodName: TypeAlias = Literal[
    "common_factor",
    "perfect_square_trinomial",
    "difference_of_squares",
    "inspection",
]

DifficultyAction: TypeAlias = Literal[
    "promoted",
    "stayed",
    "relegated",
]


DIFFICULTIES: tuple[Difficulty, ...] = (
    "easy",
    "medium",
    "hard",
    "very_hard",
)

VALID_TOPICS: tuple[MethodName, ...] = (
    "common_factor",
    "difference_of_squares",
    "perfect_square_trinomial",
    "inspection",
)


PROMOTION_ACCURACY = 0.90
RELEGATION_ACCURACY = 0.50


def update_method_difficulty(
    accuracy: float,
    current_difficulty: Difficulty,
) -> tuple[Difficulty, DifficultyAction]:
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be between 0 and 1")

    current_index = DIFFICULTIES.index(current_difficulty)

    if accuracy >= PROMOTION_ACCURACY:
        new_index = min(
            current_index + 1,
            len(DIFFICULTIES) - 1,
        )

        if new_index > current_index:
            return DIFFICULTIES[new_index], "promoted"

        # Already at very_hard.
        return current_difficulty, "stayed"

    if accuracy <= RELEGATION_ACCURACY:
        new_index = max(
            current_index - 1,
            0,
        )

        if new_index < current_index:
            return DIFFICULTIES[new_index], "relegated"

        # Already at easy.
        return current_difficulty, "stayed"

    return current_difficulty, "stayed"


def analyze_methods(
    methods: dict,
) -> dict:
    analysis = {}

    for method in VALID_TOPICS:
        if method not in methods:
            raise ValueError(f"Missing method data: {method}")

        method_data = methods[method]

        accuracy = float(method_data["accuracy"])

        current_difficulty = method_data["difficulty"]

        if current_difficulty not in DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty for {method}: " f"{current_difficulty}"
            )

        new_difficulty, action = update_method_difficulty(
            accuracy=accuracy,
            current_difficulty=current_difficulty,
        )

        analysis[method] = {
            "accuracy": accuracy,
            "previous_difficulty": (current_difficulty),
            "new_difficulty": (new_difficulty),
            "action": action,
        }

    return analysis


def get_new_difficulties(
    analysis: dict,
) -> dict[str, str]:
    return {method: result["new_difficulty"] for method, result in analysis.items()}


def get_method_actions(
    analysis: dict,
) -> dict[str, str]:
    return {method: result["action"] for method, result in analysis.items()}


def use_computed_difficulties(methods: dict) -> dict:
    """Build generation analysis from an already-computed adaptation result."""
    analysis = {}

    for method in VALID_TOPICS:
        if method not in methods:
            raise ValueError(f"Missing method data: {method}")

        method_data = methods[method]
        difficulty = method_data["difficulty"]
        action = method_data["action"]

        if difficulty not in DIFFICULTIES:
            raise ValueError(f"Invalid difficulty for {method}: {difficulty}")

        if action not in ("promoted", "stayed", "relegated"):
            raise ValueError(f"Invalid action for {method}: {action}")

        analysis[method] = {
            "new_difficulty": difficulty,
            "action": action,
        }

    return analysis
