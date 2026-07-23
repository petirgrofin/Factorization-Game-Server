import re
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