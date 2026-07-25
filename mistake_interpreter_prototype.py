from __future__ import annotations

from typing import Any, Callable

from sympy import Expr, Poly, symbols, expand, simplify, cancel
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from string import ascii_lowercase

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

ALLOWED_SYMBOLS = {letter: symbols(letter) for letter in ascii_lowercase}


class MistakeInterpretationError(ValueError):
    pass


def parse_math_expression(value: str) -> Expr:
    cleaned = value.strip()

    if not cleaned:
        raise MistakeInterpretationError(
            "Cannot parse an empty expression."
        )

    try:
        return parse_expr(
            cleaned,
            local_dict=ALLOWED_SYMBOLS,
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception as exc:
        raise MistakeInterpretationError(
            f"Could not parse expression: {value!r}"
        ) from exc


def expression_to_string(expression: Expr) -> str:
    return str(expression).replace("**", "^")


def parenthesize(expression: Expr) -> str:
    return f"({expression_to_string(expression)})"


def get_variable(expr: Expr):
    variables = sorted(
        expr.free_symbols,
        key=lambda symbol: symbol.name,
    )

    if len(variables) != 1:
        raise MistakeInterpretationError(
            "Inspection requires exactly one variable."
        )

    return variables[0]

def get_quadratic_base_monomial(expr: Expr) -> Expr:
    expanded = expand(expr)
    variables = sorted(
        expanded.free_symbols,
        key=lambda symbol: symbol.name,
    )

    if not variables:
        raise MistakeInterpretationError(
            "Inspection requires at least one variable."
        )

    polynomial = Poly(expanded, *variables)
    terms = polynomial.terms()

    if len(terms) != 3:
        raise MistakeInterpretationError(
            "Inspection requires a three-term polynomial."
        )

    # Find the constant term.
    constant_exponents = tuple(0 for _ in variables)

    exponent_terms = {
        exponents: coefficient
        for exponents, coefficient in terms
    }

    if constant_exponents not in exponent_terms:
        raise MistakeInterpretationError(
            "Inspection requires a nonzero constant term."
        )

    nonconstant_exponents = [
        exponents
        for exponents, _ in terms
        if exponents != constant_exponents
    ]

    if len(nonconstant_exponents) != 2:
        raise MistakeInterpretationError(
            "Could not identify quadratic and linear terms."
        )

    # The exponent vectors must have the form 2e and e.
    first_exponents, second_exponents = nonconstant_exponents

    if all(
        first == 2 * second
        for first, second in zip(
            first_exponents,
            second_exponents,
        )
    ):
        base_exponents = second_exponents

    elif all(
        second == 2 * first
        for first, second in zip(
            first_exponents,
            second_exponents,
        )
    ):
        base_exponents = first_exponents

    else:
        raise MistakeInterpretationError(
            "Polynomial is not quadratic in a common monomial."
        )

    base_monomial = 1

    for variable, exponent in zip(
        variables,
        base_exponents,
    ):
        base_monomial *= variable**exponent

    return base_monomial

def require_input_count(
    method: str,
    inputs: list[str],
    expected: int,
) -> None:
    if len(inputs) != expected:
        raise MistakeInterpretationError(
            f"{method} expects {expected} inputs; "
            f"received {len(inputs)}."
        )


def build_result(
    target: Expr,
    attempted_expression: Expr,
    attempted_factorization: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    expanded_attempt = expand(attempted_expression)

    return {
        "attempted_factorization": attempted_factorization,
        "expanded_attempt": expression_to_string(
            expanded_attempt
        ),
        "is_equivalent_to_target": simplify(
            expand(target) - expanded_attempt
        ) == 0,
        "diagnostics": diagnostics,
    }


def interpret_inspection(
    target: Expr,
    inputs: list[str],
) -> dict[str, Any]:
    require_input_count("inspection", inputs, 2)

    first_middle_term = parse_math_expression(inputs[0])
    second_middle_term = parse_math_expression(inputs[1])

    base = get_quadratic_base_monomial(target)

    try:
        polynomial = Poly(expand(target), base)
    except Exception as exc:
        raise ValueError(
            f"Could not interpret target as a quadratic in {base}."
        ) from exc

    if polynomial.degree() != 2:
        raise ValueError(
            "Inspection requires a quadratic expression."
        )

    # For ax² + bx + c, all_coeffs() returns [a, b, c].
    coefficients = polynomial.all_coeffs()

    if len(coefficients) != 3:
        raise ValueError(
            "Inspection requires an expression of the form "
            "a*u^2 + b*u + c."
        )

    a, b, c = coefficients

    leading_term = a * base**2
    expected_middle_term = b * base
    constant_term = c

    entered_middle_sum = expand(
        first_middle_term + second_middle_term
    )

    entered_middle_product = expand(
        first_middle_term * second_middle_term
    )

    expected_middle_product = expand(
        leading_term * constant_term
    )

    middle_sum_is_correct = (
        expand(entered_middle_sum - expected_middle_term) == 0
    )

    middle_product_is_correct = (
        expand(
            entered_middle_product - expected_middle_product
        ) == 0
    )

    # This represents the player's split of the middle term.
    attempted_expression = (
        leading_term
        + first_middle_term
        + second_middle_term
        + constant_term
    )

    result = build_result(
        target=target,
        attempted_expression=attempted_expression,
        attempted_factorization=expression_to_string(
            attempted_expression
        ),
        diagnostics={
            "base_monomial": expression_to_string(base),
            "leading_term": expression_to_string(leading_term),
            "expected_middle_term": expression_to_string(
                expected_middle_term
            ),
            "constant_term": expression_to_string(constant_term),
            "first_middle_term": expression_to_string(
                first_middle_term
            ),
            "second_middle_term": expression_to_string(
                second_middle_term
            ),
            "entered_middle_sum": expression_to_string(
                entered_middle_sum
            ),
            "middle_sum_is_correct": middle_sum_is_correct,
            "entered_middle_product": expression_to_string(
                entered_middle_product
            ),
            "expected_middle_product": expression_to_string(
                expected_middle_product
            ),
            "middle_product_is_correct": middle_product_is_correct,
        },
    )

    result["is_valid_inspection_split"] = (
        middle_sum_is_correct
        and middle_product_is_correct
    )

    return result


def interpret_difference_of_squares(
    target: Expr,
    inputs: list[str],
) -> dict[str, Any]:
    require_input_count(
        "difference_of_squares",
        inputs,
        2,
    )

    a = parse_math_expression(inputs[0])
    b = parse_math_expression(inputs[1])

    first = a - b
    second = a + b
    attempted = first * second

    return build_result(
        target=target,
        attempted_expression=attempted,
        attempted_factorization=(
            f"{parenthesize(first)}{parenthesize(second)}"
        ),
        diagnostics={
            "first_square_root": expression_to_string(a),
            "second_square_root": expression_to_string(b),
            "constructed_difference": expression_to_string(
                expand(a**2 - b**2)
            ),
        },
    )


def interpret_perfect_square_trinomial(
    target: Expr,
    inputs: list[str],
) -> dict[str, Any]:
    require_input_count(
        "perfect_square_trinomial",
        inputs,
        2,
    )

    a = parse_math_expression(inputs[0])
    b = parse_math_expression(inputs[1])

    binomial = a + b
    attempted = binomial**2

    return build_result(
        target=target,
        attempted_expression=attempted,
        attempted_factorization=(
            f"{parenthesize(binomial)}^2"
        ),
        diagnostics={
            "first_binomial_term": expression_to_string(a),
            "second_binomial_term": expression_to_string(b),
            "constructed_first_term": expression_to_string(a**2),
            "constructed_middle_term": expression_to_string(
                2 * a * b
            ),
            "constructed_last_term": expression_to_string(b**2),
        },
    )


def interpret_common_factor(
    target: Expr,
    inputs: list[str],
) -> dict[str, Any]:
    require_input_count("common_factor", inputs, 1)

    extracted = parse_math_expression(inputs[0])

    if extracted == 0:
        raise MistakeInterpretationError(
            "Common factor cannot be zero."
        )

    quotient = cancel(target / extracted)

    variables = sorted(
        target.free_symbols | extracted.free_symbols,
        key=lambda symbol: symbol.name,
    )

    try:
        quotient_polynomial = (
            quotient.as_poly(*variables)
            if variables
            else quotient.as_poly()
        )
    except Exception:
        quotient_polynomial = None

    divides_evenly = (
        quotient_polynomial is not None
        and all(
            coefficient.is_integer is True
            for coefficient in quotient_polynomial.coeffs()
        )
    )

    result = build_result(
        target=target,
        attempted_expression=extracted * quotient,
        attempted_factorization=(
            f"{parenthesize(extracted)}"
            f"{parenthesize(quotient)}"
        ),
        diagnostics={
            "extracted_factor": expression_to_string(extracted),
            "computed_quotient": expression_to_string(quotient),
            "factor_divides_polynomial_evenly": divides_evenly,
        },
    )

    result["is_valid_factorization"] = divides_evenly

    return result


Interpreter = Callable[
    [Expr, list[str]],
    dict[str, Any],
]


INTERPRETERS: dict[str, Interpreter] = {
    "common_factor": interpret_common_factor,
    "perfect_square_trinomial": (
        interpret_perfect_square_trinomial
    ),
    "difference_of_squares": interpret_difference_of_squares,
    "inspection": interpret_inspection,
}


def interpret_polynomial_mistake(
    mistake: dict[str, Any],
) -> dict[str, Any]:
    method = mistake["method"]
    interpreter = INTERPRETERS.get(method)

    if interpreter is None:
        raise MistakeInterpretationError(
            f"Unsupported method: {method!r}"
        )

    target = parse_math_expression(
        mistake["target_factor"]
    )

    interpretation = interpreter(
        target,
        mistake.get("inputs", []),
    )

    return {
        **mistake,
        "interpretation": interpretation,
    }


def enrich_mistakes(
    mistakes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = []

    for mistake in mistakes:
        if mistake.get("type") != "polynomial":
            enriched.append(mistake)
            continue

        try:
            enriched.append(
                interpret_polynomial_mistake(mistake)
            )
        except MistakeInterpretationError as exc:
            enriched.append({
                **mistake,
                "interpretation": {
                    "attempted_factorization": None,
                    "expanded_attempt": None,
                    "is_equivalent_to_target": None,
                    "diagnostics": {},
                    "interpretation_error": str(exc),
                },
            })

    return enriched