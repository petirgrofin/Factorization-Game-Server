from __future__ import annotations

from math import gcd, isqrt
from functools import reduce
from string import ascii_lowercase
from typing import Any, Callable

from sympy import Expr, Poly, cancel, expand, simplify, symbols
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

ALLOWED_SYMBOLS = {
    letter: symbols(letter)
    for letter in ascii_lowercase
}


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


def expressions_are_equal(first: Expr, second: Expr) -> bool:
    return simplify(expand(first) - expand(second)) == 0


def monomial_parts(expression: Expr) -> tuple[int, dict[Expr, int]]:
    """Return the integer coefficient and variable exponents of a monomial."""
    expanded = expand(expression)
    coefficient, powers = expanded.as_coeff_Mul()

    if coefficient.is_integer is not True:
        raise MistakeInterpretationError(
            f"{expression_to_string(expression)} must have an integer coefficient."
        )

    variable_powers: dict[Expr, int] = {}

    if powers == 1:
        return int(coefficient), variable_powers

    for factor in powers.as_ordered_factors():
        base, exponent = factor.as_base_exp()

        if not base.is_Symbol or exponent.is_integer is not True or exponent < 0:
            raise MistakeInterpretationError(
                f"{expression_to_string(expression)} must be a monomial."
            )

        variable_powers[base] = variable_powers.get(base, 0) + int(exponent)

    return int(coefficient), variable_powers


def monomial_from_parts(
    coefficient: int,
    variable_powers: dict[Expr, int],
) -> Expr:
    result: Expr = coefficient

    for variable in sorted(variable_powers, key=lambda symbol: symbol.name):
        result *= variable ** variable_powers[variable]

    return result


def greatest_common_monomial(target: Expr) -> Expr:
    expanded = expand(target)
    variables = sorted(expanded.free_symbols, key=lambda symbol: symbol.name)
    polynomial = Poly(expanded, *variables) if variables else Poly(expanded)
    terms = polynomial.terms()

    if not terms:
        raise MistakeInterpretationError("Cannot find a common factor of zero.")

    numeric_gcf = reduce(gcd, (abs(int(coefficient)) for _, coefficient in terms))
    variable_powers = {
        variable: min(exponents[index] for exponents, _ in terms)
        for index, variable in enumerate(variables)
    }

    return monomial_from_parts(numeric_gcf, variable_powers)


def monomial_divides(divisor: Expr, target: Expr) -> bool:
    if divisor == 0:
        return False

    variables = sorted(
        divisor.free_symbols | target.free_symbols,
        key=lambda symbol: symbol.name,
    )
    quotient = cancel(target / divisor)
    polynomial = quotient.as_poly(*variables) if variables else quotient.as_poly()

    return (
        polynomial is not None
        and all(coefficient.is_integer is True for coefficient in polynomial.coeffs())
    )


def positive_monomial_square_root(expression: Expr) -> Expr:
    coefficient, powers = monomial_parts(expression)

    if coefficient < 0:
        raise MistakeInterpretationError(
            f"{expression_to_string(expression)} is not a positive square monomial."
        )

    coefficient_root = isqrt(coefficient)

    if coefficient_root**2 != coefficient or any(
        exponent % 2 != 0 for exponent in powers.values()
    ):
        raise MistakeInterpretationError(
            f"{expression_to_string(expression)} is not a perfect-square monomial."
        )

    return monomial_from_parts(
        coefficient_root,
        {variable: exponent // 2 for variable, exponent in powers.items()},
    )


def compare_monomial_components(
    entered: Expr,
    expected: Expr,
) -> tuple[bool, bool]:
    entered_coefficient, entered_powers = monomial_parts(entered)
    expected_coefficient, expected_powers = monomial_parts(expected)

    numeric_correct = abs(entered_coefficient) == abs(expected_coefficient)
    variables_correct = entered_powers == expected_powers
    return numeric_correct, variables_correct


def interpret_inspection(
    target: Expr,
    inputs: list[str],
) -> dict[str, str | None]:
    """Interpret the four boxes used by the inspection method.

    For a quadratic ax^2 + bx + c:
      - input 1 * input 2 must equal ax^2;
      - input 3 * input 4 must equal c;
      - input 1 * input 4 + input 2 * input 3 must equal bx.

    The four inputs therefore represent:
        (input 1 + input 3)(input 2 + input 4)
    """
    require_input_count("inspection", inputs, 4)

    first, second, third, fourth = (
        parse_math_expression(value)
        for value in inputs
    )

    first_factor = first + third
    second_factor = second + fourth
    attempted_factorization = (
        f"{parenthesize(first_factor)}"
        f"{parenthesize(second_factor)}"
    )

    variables = sorted(
        target.free_symbols,
        key=lambda symbol: symbol.name,
    )

    if len(variables) != 1:
        raise MistakeInterpretationError(
            "Inspection requires a quadratic in exactly one variable."
        )

    variable = variables[0]

    try:
        polynomial = Poly(expand(target), variable)
    except Exception as exc:
        raise MistakeInterpretationError(
            "Inspection requires a polynomial expression."
        ) from exc

    if polynomial.degree() != 2:
        raise MistakeInterpretationError(
            "Inspection requires a quadratic expression."
        )

    a, b, c = polynomial.all_coeffs()
    expected_leading = a * variable**2
    expected_middle = b * variable

    entered_leading = expand(first * second)
    entered_constant = expand(third * fourth)
    entered_middle = expand(
        first * fourth + second * third
    )

    errors: list[str] = []

    if not expressions_are_equal(
        entered_leading,
        expected_leading,
    ):
        errors.append(
            f"the first two inputs multiply to "
            f"{expression_to_string(entered_leading)}, not "
            f"{expression_to_string(expected_leading)}"
        )

    if not expressions_are_equal(entered_constant, c):
        errors.append(
            f"the last two inputs multiply to "
            f"{expression_to_string(entered_constant)}, not "
            f"{expression_to_string(c)}"
        )

    if not expressions_are_equal(entered_middle, expected_middle):
        errors.append(
            f"the cross-products add to "
            f"{expression_to_string(entered_middle)}, not "
            f"{expression_to_string(expected_middle)}"
        )

    return {
        "attempted_factorization": attempted_factorization,
        "reason": (
            None
            if not errors
            else "The inspection is incorrect because "
            + "; ".join(errors)
            + "."
        ),
    }


def interpret_difference_of_squares(
    target: Expr,
    inputs: list[str],
) -> dict[str, str | None]:
    require_input_count("difference_of_squares", inputs, 2)

    first_root = parse_math_expression(inputs[0])
    second_root = parse_math_expression(inputs[1])

    first_factor = first_root - second_root
    second_factor = first_root + second_root
    attempted_factorization = (
        f"{parenthesize(first_factor)}"
        f"{parenthesize(second_factor)}"
    )

    expanded_target = expand(target)
    terms = expanded_target.as_ordered_terms()

    if len(terms) != 2:
        raise MistakeInterpretationError(
            "Difference of squares requires exactly two terms."
        )

    positive_terms = [term for term in terms if term.as_coeff_Mul()[0] > 0]
    negative_terms = [term for term in terms if term.as_coeff_Mul()[0] < 0]

    if len(positive_terms) != 1 or len(negative_terms) != 1:
        raise MistakeInterpretationError(
            "Difference of squares requires one positive term and one negative term."
        )

    expected_first_root = positive_monomial_square_root(positive_terms[0])
    expected_second_root = positive_monomial_square_root(-negative_terms[0])

    if expressions_are_equal(first_root, expected_first_root) and expressions_are_equal(
        second_root, expected_second_root
    ):
        return {
            "attempted_factorization": attempted_factorization,
            "reason": None,
        }

    first_numeric, first_variables = compare_monomial_components(
        first_root, expected_first_root
    )
    second_numeric, second_variables = compare_monomial_components(
        second_root, expected_second_root
    )

    numeric_correct = first_numeric and second_numeric
    variables_correct = first_variables and second_variables

    if numeric_correct and not variables_correct:
        reason = (
            "The numeric parts of the square roots are correct, but the "
            "variable parts are not. The expected roots are "
            f"{expression_to_string(expected_first_root)} and "
            f"{expression_to_string(expected_second_root)}."
        )
    elif variables_correct and not numeric_correct:
        reason = (
            "The variable parts of the square roots are correct, but the "
            "numeric parts are not. The expected roots are "
            f"{expression_to_string(expected_first_root)} and "
            f"{expression_to_string(expected_second_root)}."
        )
    else:
        reason = (
            "Neither the numeric parts nor the variable parts of the entered "
            "square roots match the target. The expected roots are "
            f"{expression_to_string(expected_first_root)} and "
            f"{expression_to_string(expected_second_root)}."
        )

    return {
        "attempted_factorization": attempted_factorization,
        "reason": reason,
    }


def interpret_perfect_square_trinomial(
    target: Expr,
    inputs: list[str],
) -> dict[str, str | None]:
    require_input_count(
        "perfect_square_trinomial",
        inputs,
        2,
    )

    first_term = parse_math_expression(inputs[0])
    second_term = parse_math_expression(inputs[1])

    binomial = first_term + second_term
    attempted_factorization = f"{parenthesize(binomial)}^2"

    expanded_target = expand(target)
    terms = expanded_target.as_ordered_terms()

    if len(terms) != 3:
        raise MistakeInterpretationError(
            "A perfect-square trinomial requires exactly three terms."
        )

    square_candidates: list[tuple[Expr, Expr]] = []

    for term in terms:
        try:
            root = positive_monomial_square_root(term)
        except MistakeInterpretationError:
            continue
        square_candidates.append((term, root))

    if len(square_candidates) != 2:
        raise MistakeInterpretationError(
            "Could not identify two positive perfect-square terms in the target."
        )

    first_square, expected_first = square_candidates[0]
    second_square, unsigned_second = square_candidates[1]
    middle_terms = [
        term
        for term in terms
        if term != first_square and term != second_square
    ]

    if len(middle_terms) != 1:
        raise MistakeInterpretationError(
            "Could not identify the middle term of the trinomial."
        )

    middle_term = middle_terms[0]
    positive_middle = expand(2 * expected_first * unsigned_second)
    negative_middle = expand(-positive_middle)

    if expressions_are_equal(middle_term, positive_middle):
        expected_second = unsigned_second
    elif expressions_are_equal(middle_term, negative_middle):
        expected_second = -unsigned_second
    else:
        raise MistakeInterpretationError(
            "The target is not a perfect-square trinomial: its middle term "
            "is not twice the product of the square roots of the outer terms."
        )

    if expressions_are_equal(first_term, expected_first) and expressions_are_equal(
        second_term, expected_second
    ):
        return {
            "attempted_factorization": attempted_factorization,
            "reason": None,
        }

    first_numeric, first_variables = compare_monomial_components(
        first_term, expected_first
    )
    second_numeric, second_variables = compare_monomial_components(
        second_term, expected_second
    )

    numeric_correct = first_numeric and second_numeric
    variables_correct = first_variables and second_variables
    entered_first_coefficient, _ = monomial_parts(first_term)
    expected_first_coefficient, _ = monomial_parts(expected_first)
    entered_second_coefficient, _ = monomial_parts(second_term)
    expected_second_coefficient, _ = monomial_parts(expected_second)

    signs_correct = (
        (entered_first_coefficient < 0) == (expected_first_coefficient < 0)
        and (entered_second_coefficient < 0) == (expected_second_coefficient < 0)
    )

    expected_binomial = expected_first + expected_second
    expected_text = f"{parenthesize(expected_binomial)}^2"

    if numeric_correct and variables_correct and not signs_correct:
        reason = (
            "The numeric and variable parts of the binomial terms are correct, "
            "but a sign is wrong. The expected factorization is "
            f"{expected_text}."
        )
    elif numeric_correct and not variables_correct:
        reason = (
            "The numeric parts of the binomial terms are correct, but the "
            "variable parts are not. The expected factorization is "
            f"{expected_text}."
        )
    elif variables_correct and not numeric_correct:
        reason = (
            "The variable parts of the binomial terms are correct, but the "
            "numeric parts are not. The expected factorization is "
            f"{expected_text}."
        )
    else:
        reason = (
            "Neither the numeric parts nor the variable parts of the entered "
            "binomial terms match the target. The expected factorization is "
            f"{expected_text}."
        )

    return {
        "attempted_factorization": attempted_factorization,
        "reason": reason,
    }


def interpret_common_factor(
    target: Expr,
    inputs: list[str],
) -> dict[str, str | None]:
    require_input_count("common_factor", inputs, 1)

    extracted = parse_math_expression(inputs[0])

    if extracted == 0:
        raise MistakeInterpretationError("Common factor cannot be zero.")

    expected_gcf = greatest_common_monomial(target)
    quotient = cancel(target / extracted)
    attempted_factorization = (
        f"{parenthesize(extracted)}"
        f"{parenthesize(quotient)}"
    )

    numeric_correct, variables_correct = compare_monomial_components(
        extracted, expected_gcf
    )

    if numeric_correct and variables_correct:
        return {
            "attempted_factorization": attempted_factorization,
            "reason": None,
        }

    if monomial_divides(extracted, target):
        return {
            "attempted_factorization": attempted_factorization,
            "reason": (
                f"{expression_to_string(extracted)} is a common factor, but it "
                "is not the greatest common factor. The greatest common factor "
                f"is {expression_to_string(expected_gcf)}."
            ),
        }

    if numeric_correct and not variables_correct:
        reason = (
            "The numeric part of the entered common factor is correct, but "
            "the variable part is not. The greatest common factor is "
            f"{expression_to_string(expected_gcf)}."
        )
    elif variables_correct and not numeric_correct:
        reason = (
            "The variable part of the entered common factor is correct, but "
            "the numeric part is not. The greatest common factor is "
            f"{expression_to_string(expected_gcf)}."
        )
    else:
        reason = (
            "Neither the numeric part nor the variable part of the entered "
            "common factor is correct. The greatest common factor is "
            f"{expression_to_string(expected_gcf)}."
        )

    return {
        "attempted_factorization": attempted_factorization,
        "reason": reason,
    }


Interpreter = Callable[
    [Expr, list[str]],
    dict[str, str | None],
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

    target = parse_math_expression(mistake["target_factor"])

    return {
        **mistake,
        "interpretation": interpreter(
            target,
            mistake.get("inputs", []),
        ),
    }


def enrich_mistakes(
    mistakes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []

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
                    "reason": str(exc),
                },
            })

    return enriched