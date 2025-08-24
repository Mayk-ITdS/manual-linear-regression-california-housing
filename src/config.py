from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

PROCESSED_DIR = DATA_DIR / "processed"
SUMMARY_DIR   = DATA_DIR / "summary"
MODELS_DIR    = DATA_DIR / "models"
LOGS_DIR      = DATA_DIR / "logs"
PLOTS_DIR     = DATA_DIR / "plots"

for d in [PROCESSED_DIR, SUMMARY_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAINING_CONFIG = {
    "normal": {
        "ols":   {"lambda": 0.0},
        "ridge": {"lambda": 100}
    },
    "gd": {
        "ols": {
            "learning_rate": 0.01,
            "epochs": 50000,
            "ridge_lambda": 0.0,
            "patience": 100
        },
        "ridge": {
            "learning_rate": 0.01,
            "epochs": 50000,
            "ridge_lambda": 100,
            "patience": 100
        }
    }
}

VIS_CONFIG = {
    "style": "seaborn-v0_8",
    "figsize": (10, 6),
    "dpi": 120,
    "colors": {
        "train": "royalblue",
        "val": "darkorange",
        "mse": "crimson",
        "r2": "forestgreen"
    }
}
