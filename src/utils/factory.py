from models.regression import LinearRegression, NormalSolver, GradientSolver
from config import TRAINING_CONFIG

SOLVER_NAMES = {
    1: "OLS_NE",
    2: "Ridge_NE",
    3: "OLS_GD",
    4: "Ridge_GD"
}

def solver_name(choice: int) -> str:
    return SOLVER_NAMES.get(choice, "Unknown")

def make_regressor(choice: int) -> LinearRegression:

    if choice == 1:
        solver = NormalSolver(lambda_=TRAINING_CONFIG["normal"]["ols"]["lambda"])
    elif choice == 2:
        solver = NormalSolver(lambda_=TRAINING_CONFIG["normal"]["ridge"]["lambda"])
    elif choice == 3:
        solver = GradientSolver(**TRAINING_CONFIG["gd"]["ols"])
    elif choice == 4:
        solver = GradientSolver(**TRAINING_CONFIG["gd"]["ridge"])
    else:
        raise ValueError(f"Unknown solver choice: {choice}")

    return LinearRegression(solver)
