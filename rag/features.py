import re

from sympy import symbols, sympify, Poly, factor_list

x, y, z, a, b, c = symbols("x y z a b c")

def polynomial_description(expr_str: str):

    expr_str = expr_str.replace("^", "**")

    expr = sympify(expr_str)

    poly = Poly(expr)

    degree = poly.total_degree()

    variables = len(poly.gens)

    num_terms = len(poly.terms())

    description = []

    description.append(f"Polynomial with {num_terms} terms.")
    description.append(f"Total degree {degree}.")
    description.append(f"{variables} variables.")

    # Difference of squares heuristic
    if num_terms == 2:
        coeffs = poly.coeffs()

        if coeffs[0] * coeffs[1] < 0:
            description.append("Contains subtraction between terms.")

            exponents_even = True

            for monomial in poly.monoms():
                for exp in monomial:
                    if exp % 2 != 0:
                        exponents_even = False

            if exponents_even:
                description.append("Difference of squares structure.")

    # Perfect square trinomial heuristic
    if num_terms == 3:
        description.append("Trinomial structure.")

    # Multi-step factoring heuristic
    factors = factor_list(expr)

    if len(factors[1]) > 1:
        description.append("Can be factored into multiple factors.")

    return " ".join(description)