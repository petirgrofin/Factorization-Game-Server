from sympy import symbols, sympify, factor
from server.utils import to_sympy_format
x = symbols('x')

def validate_polynomial(exercise_text):
    try:
        expr = sympify(to_sympy_format(exercise_text))

        factored = factor(expr)

        # If factoring changes the expression,
        # then it is factorisable over the integers
        return factored != expr

    except Exception as e:
        print("Validation error:", e)
        return False