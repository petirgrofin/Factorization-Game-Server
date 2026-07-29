from math import floor

from server.difficulty import MethodName

TOTAL_EXERCISES = 6

ACTION_WEIGHTS = {
    "relegated": 3.0,
    "stayed": 2.0,
    "promoted": 1.0,
}

ACTION_CATEGORIES = {
    "relegated": "focus",
    "stayed": "focus",
    "promoted": "review",
}


def allocate_counts(
    weighted_methods: list[tuple[MethodName, float]],
    total: int,
    guarantee_each: bool = True,
) -> dict[MethodName, int]:
    """
    Allocate `total` exercises proportionally according
    to each method's weight.

    If guarantee_each is True and there are enough exercises,
    every method receives at least one exercise.
    """
    if total <= 0 or not weighted_methods:
        return {}

    if any(weight <= 0 for _, weight in weighted_methods):
        raise ValueError("All allocation weights must be positive.")

    method_count = len(weighted_methods)

    allocated: dict[MethodName, int] = {method: 0 for method, _ in weighted_methods}

    remaining = total

    # Give every method one exercise when possible.
    if guarantee_each and total >= method_count:
        for method, _ in weighted_methods:
            allocated[method] = 1

        remaining -= method_count

    if remaining == 0:
        return allocated

    total_weight = sum(weight for _, weight in weighted_methods)

    exact_additional_counts = {
        method: remaining * weight / total_weight for method, weight in weighted_methods
    }

    floor_additional_counts = {
        method: floor(exact_count)
        for method, exact_count in exact_additional_counts.items()
    }

    for method, count in floor_additional_counts.items():
        allocated[method] += count

    exercises_left = remaining - sum(floor_additional_counts.values())

    # Largest-remainder method.
    #
    # The second key makes ties deterministic:
    # methods earlier in weighted_methods win the tie.
    original_order = {
        method: index for index, (method, _) in enumerate(weighted_methods)
    }

    ordered_methods = sorted(
        exact_additional_counts,
        key=lambda method: (
            -(exact_additional_counts[method] - floor_additional_counts[method]),
            original_order[method],
        ),
    )

    for method in ordered_methods[:exercises_left]:
        allocated[method] += 1

    return {method: count for method, count in allocated.items() if count > 0}


def create_exercise_plan(
    analysis: dict,
    total_exercises: int = TOTAL_EXERCISES,
) -> list[dict]:
    weighted_methods: list[tuple[MethodName, float]] = []

    for method, result in analysis.items():
        action = result["action"]

        if action not in ACTION_WEIGHTS:
            raise ValueError(f"Unknown action for {method}: {action}")

        base_weight = ACTION_WEIGHTS[action]
        accuracy = result.get("accuracy")
        weight = base_weight

        # Preserve the previous fine-grained weighting when an analysis contains
        # accuracy, while allowing generation from a precomputed difficulty/action.
        if accuracy is not None:
            weight += 0.25 * (1.0 - accuracy)

        weighted_methods.append(
            (
                method,
                weight,
            )
        )

    allocation = allocate_counts(
        weighted_methods=weighted_methods,
        total=total_exercises,
        guarantee_each=True,
    )

    plan: list[dict] = []

    for method, count in allocation.items():
        action = analysis[method]["action"]

        plan.append(
            {
                "method": method,
                "difficulty": analysis[method]["new_difficulty"],
                "count": count,
                "category": ACTION_CATEGORIES[action],
            }
        )

    return plan
