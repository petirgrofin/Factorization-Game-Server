def format_player_inputs(inputs: list) -> str:
    if not inputs:
        return "No inputs provided"

    return ", ".join(str(value) for value in inputs)


def format_polynomial_mistake(mistake: dict) -> str:
    inputs = mistake.get("inputs", [])
    formatted_inputs = format_player_inputs(inputs)

    interpretation = mistake.get("interpretation") or {}

    attempted = interpretation.get(
        "attempted_factorization"
    )
    if not attempted:
        attempted = "Could not be interpreted"

    reason = interpretation.get("reason")
    if not reason:
        reason = "No error reason was provided"

    original_polynomial = mistake.get(
        "original_polynomial",
        "Not provided",
    )

    current_factorization = mistake.get(
        "current_factorization",
        "Not provided",
    )

    target_factor = mistake.get(
        "target_factor",
        "Not provided",
    )

    method = mistake.get(
        "method",
        "Not provided",
    )

    return (
        "Mistake type: polynomial factorization\n"
        f"Original polynomial: {original_polynomial}\n"
        f"Current factorization: {current_factorization}\n"
        f"Selected target: {target_factor}\n"
        f"Method used: {method}\n"
        f"Player inputs: {formatted_inputs}\n"
        f"Interpreted attempt: {attempted}\n"
        f"Reason the attempt is incorrect: {reason}"
    )


def format_question_mistake(mistake: dict) -> str:
    player_response = mistake.get("player_response")

    if player_response is True:
        response = "yes"
    elif player_response is False:
        response = "no"
    else:
        response = "Not provided"

    correct_response = mistake.get("correct_response")

    if correct_response is True:
        expected = "yes"
    elif correct_response is False:
        expected = "no"
    else:
        expected = "Not provided"

    question_text = mistake.get(
        "question_text",
        "Not provided",
    )

    return (
        "Mistake type: conceptual question\n"
        f"Question: {question_text}\n"
        f"Player response: {response}\n"
        f"Correct response: {expected}"
    )


def format_mistakes_for_prompt(
    mistakes: list[dict],
) -> str:
    if not mistakes:
        return "No recent mistake details are available."

    formatted = []

    for index, mistake in enumerate(
        mistakes,
        start=1,
    ):
        mistake_type = mistake.get("type")

        if mistake_type == "polynomial":
            description = format_polynomial_mistake(
                mistake
            )
        elif mistake_type == "question":
            description = format_question_mistake(
                mistake
            )
        else:
            continue

        formatted.append(
            f"Mistake {index}:\n{description}"
        )

    if not formatted:
        return "No recognizable mistake details are available."

    return "\n\n".join(formatted)


def format_mistakes_for_retrieval(
    mistakes: list[dict],
) -> str:
    if not mistakes:
        return "No recent mistakes"

    descriptions = []

    for mistake in mistakes:
        mistake_type = mistake.get("type")

        if mistake_type == "polynomial":
            interpretation = (
                mistake.get("interpretation") or {}
            )

            attempted = interpretation.get(
                "attempted_factorization"
            )
            reason = interpretation.get("reason")

            descriptions.append(
                "Polynomial mistake: "
                f"method={mistake.get('method')}; "
                f"original="
                f"{mistake.get('original_polynomial')}; "
                f"current_factorization="
                f"{mistake.get('current_factorization')}; "
                f"target={mistake.get('target_factor')}; "
                f"inputs={mistake.get('inputs', [])}; "
                f"attempt={attempted}; "
                f"reason={reason}"
            )

        elif mistake_type == "question":
            descriptions.append(
                "Question mistake: "
                f"question="
                f"{mistake.get('question_text')}; "
                f"player_response="
                f"{mistake.get('player_response')}; "
                f"correct_response="
                f"{mistake.get('correct_response')}"
            )

    if not descriptions:
        return "No recognizable recent mistakes"

    return "\n".join(descriptions)