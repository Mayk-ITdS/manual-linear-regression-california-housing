import numpy as np
import pandas as pd
from datetime import datetime
from utils.factory import make_regressor, solver_name
from config import PROCESSED_DIR, MODELS_DIR, LOGS_DIR


def main():
    while True:
        print("""
        =================================================
             Welcome to Majk's Linear Regression Lab
              (OLS & Ridge — Normal Eq. / Gradient)
        =================================================
        """)

        print("=== Available Solvers ===")
        print("  1. OLS (Normal Equation)")
        print("  2. Ridge (Normal Equation)")
        print("  3. OLS (Gradient Descent)")
        print("  4. Ridge (Gradient Descent)")

        choice = int(input("\nChoose solver (1-4): "))
        regressor = make_regressor(choice)
        solver_label = solver_name(choice)

        df = pd.read_csv(PROCESSED_DIR / "df_zscore_final.csv")
        X = df.drop("MedHouseVal",axis=1).to_numpy(dtype=float)
        y= df["MedHouseVal"].to_numpy(dtype=float)

        print("X mean:", X.mean(axis=0))
        print("X std:", X.std(axis=0))

        regressor.fit(X, y)
        r2 = regressor.score(X, y)
        print(f"\nTraining complete using {solver_label}. R² = {r2:.4f}")

        timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        model_file = MODELS_DIR / f"model_{solver_label}_{timestamp}.csv"
        history_file = MODELS_DIR / f"history_{solver_label}_{timestamp}.csv"
        log_file = LOGS_DIR / f"training_{solver_label}_{timestamp}.log"

        if hasattr(regressor.solver, "save_best_model"):
            regressor.solver.save_best_model(filepath=model_file)
        if hasattr(regressor.solver, "save_training_history"):
            regressor.solver.save_training_history(filepath=history_file)

        with open(log_file, "w",encoding="utf-8") as f:
            f.write(f"Training completed at {timestamp}\n")
            f.write(f"Solver: {solver_label}\n")
            f.write(f"R²: {r2:.4f}\n")

        print("\nFiles created:")
        print(f"   • {model_file.name} — Best model (weights, bias, metrics)")
        print(f"   • {history_file.name} — Training history (per epoch)")
        print(f"   • {log_file.name} — Training log")
        c = input(f"\nWould You like to try another model or change parameters?[Y/N]")
        if c.lower() == "y":
            continue
        else:
            print("Thank You for using my program!")
            print("See you next time!\n Michal")
            break


if __name__ == "__main__":
    main()
