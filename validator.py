import re
from sympy import symbols, sympify, factor

x = symbols('x')

def to_sympy_format(expr_str):
    # Replace ^ with **
    expr_str = expr_str.replace("^", "**")

    # Insert * between numbers and variables
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)

    # Insert * between variables and parentheses
    expr_str = re.sub(r'([a-zA-Z])\(', r'\1*(', expr_str)

    # Insert * between closing/opening parentheses
    expr_str = re.sub(r'\)\(', r')*(', expr_str)

    return expr_str

def validate_exercise(exercise_text):
    try:
        expr = sympify(to_sympy_format(exercise_text))

        factored = factor(expr)

        # If factoring changes the expression,
        # then it is factorisable over the integers
        return factored != expr

    except Exception as e:
        print("Validation error:", e)
        return False