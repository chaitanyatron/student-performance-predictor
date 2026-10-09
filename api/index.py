"""Student Performance Predictor - Flask backend (works locally and on Vercel).

Vercel looks for a Flask object named `app` inside api/index.py.
Local run:  python api/index.py
"""
import math
from pathlib import Path

import pandas as pd
import sklearn
from flask import Flask, jsonify, request, send_from_directory
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Paths: built from this file's location, so they work locally and on Vercel.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "students.csv"
PUBLIC_DIR = BASE_DIR / "public"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LABELS = ["Low", "Average", "High"]                      # fixed display order
PARTICIPATION_MAP = {"Low": 1, "Medium": 2, "High": 3}   # text -> number
NUMERIC_RANGES = {                                       # feature: (min, max)
    "Attendance_Percentage": (0, 100),
    "Study_Hours_Per_Day": (0, 24),
    "Previous_Marks": (0, 100),
    "Assignment_Completion_Percentage": (0, 100),
    "Sleep_Hours": (0, 24),
}
FEATURES = list(NUMERIC_RANGES) + ["Participation_Level"]
REQUIRED_COLUMNS = ["Student_ID"] + FEATURES + ["Performance"]
NICE_NAMES = {
    "Attendance_Percentage": "Attendance",
    "Study_Hours_Per_Day": "Study hours per day",
    "Previous_Marks": "Previous marks",
    "Assignment_Completion_Percentage": "Assignment completion",
    "Sleep_Hours": "Sleep hours",
    "Participation_Level": "Participation level",
}
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_DEPTH = 4


# ---------------------------------------------------------------------------
# 1. Load and validate the CSV with pandas
# ---------------------------------------------------------------------------
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("CSV contains empty (missing) values")
    if not set(df["Performance"]).issubset(LABELS):
        raise ValueError(f"Performance must be one of {LABELS}")
    if not set(df["Participation_Level"]).issubset(PARTICIPATION_MAP):
        raise ValueError(f"Participation_Level must be one of {list(PARTICIPATION_MAP)}")
    for col in NUMERIC_RANGES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")
    return df


def make_features(df):
    """Turn a DataFrame into model input (Participation_Level text -> 1/2/3)."""
    X = df[FEATURES].copy()
    X["Participation_Level"] = X["Participation_Level"].map(PARTICIPATION_MAP)
    return X


# ---------------------------------------------------------------------------
# 2. Train and evaluate ONCE when the server starts (cached in module variables)
# ---------------------------------------------------------------------------
def build_model():
    df = load_dataset()
    X = make_features(df)
    y = df["Performance"]

    # Split FIRST so the test students are never seen during training.
    # (The only preprocessing is a fixed text->number mapping that learns nothing
    #  from the data, so there is no fitted scaler/encoder that could leak.)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    # Metrics come from predictions on the held-out test set only.
    y_pred = model.predict(X_test)
    metrics = {
        "dataset_size": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "class_distribution": {k: int((y == k).sum()) for k in LABELS},
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "labels": LABELS,
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=LABELS).tolist(),
        "confusion_matrix_note": "Rows = actual class, columns = predicted class (held-out test set).",
        "feature_importances": {
            NICE_NAMES[f]: round(float(v), 4) for f, v in zip(FEATURES, model.feature_importances_)
        },
        "tree_depth": int(model.get_depth()),
        "model": f"DecisionTreeClassifier(max_depth={MAX_DEPTH}, random_state={RANDOM_STATE})",
    }
    return df, model, metrics


try:
    DF, MODEL, METRICS = build_model()
    STARTUP_ERROR = None
except Exception as exc:  # keep the server alive so /api/health can report the problem
    DF, MODEL, METRICS = None, None, None
    STARTUP_ERROR = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# 3. Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)  # `app` is the WSGI entry point Vercel uses
app.json.sort_keys = False


def error(message, status, details=None):
    body = {"error": message}
    if details:
        body["details"] = details
    return jsonify(body), status


def model_unavailable():
    return error("Model is not available", 500, [STARTUP_ERROR])


@app.get("/api/health")
def health():
    if STARTUP_ERROR:
        return jsonify(status="error", model_trained=False, message=STARTUP_ERROR), 503
    return jsonify(
        status="ok",
        model_trained=True,
        dataset_rows=int(len(DF)),
        model="DecisionTreeClassifier",
        scikit_learn_version=sklearn.__version__,
    )


@app.get("/api/metrics")
def metrics():
    if STARTUP_ERROR:
        return model_unavailable()
    return jsonify(METRICS)


@app.get("/api/students")
def students():
    """First rows of the dataset for the preview table (?limit=1..50)."""
    if STARTUP_ERROR:
        return model_unavailable()
    try:
        limit = min(max(int(request.args.get("limit", 10)), 1), 50)
    except ValueError:
        return error("limit must be a whole number", 400)
    return jsonify(columns=REQUIRED_COLUMNS, rows=DF.head(limit).to_dict(orient="records"))


def validate_payload(data):
    """Return (clean_values, list_of_problems)."""
    problems, clean = [], {}
    for name, (low, high) in NUMERIC_RANGES.items():
        if name not in data:
            problems.append(f"{name} is required")
            continue
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            problems.append(f"{name} must be a number")
            continue
        try:
            value = float(value)
        except ValueError:
            problems.append(f"{name} must be a number")
            continue
        if math.isnan(value) or math.isinf(value):
            problems.append(f"{name} must be a finite number")
        elif not low <= value <= high:
            problems.append(f"{name} must be between {low} and {high}")
        else:
            clean[name] = value

    level = data.get("Participation_Level")
    if level is None:
        problems.append("Participation_Level is required")
    elif not isinstance(level, str) or level.strip().capitalize() not in PARTICIPATION_MAP:
        problems.append("Participation_Level must be one of: Low, Medium, High")
    else:
        clean["Participation_Level"] = level.strip().capitalize()
    return clean, problems


def explain(row, predicted):
    """Describe the decision-tree rules this student followed."""
    tree = MODEL.tree_
    path = MODEL.decision_path(row).indices
    reasons = []
    for node in path:
        if tree.children_left[node] == tree.children_right[node]:  # leaf node
            continue
        feature = FEATURES[tree.feature[node]]
        threshold = float(tree.threshold[node])
        value = float(row.iloc[0][feature])
        goes_left = value <= threshold
        if feature == "Participation_Level":
            names = list(PARTICIPATION_MAP)
            chosen = names[: int(threshold)] if goes_left else names[int(threshold):]
            reasons.append(f"participation level is {' or '.join(chosen)}")
        else:
            word = "at most" if goes_left else "above"
            reasons.append(f"{NICE_NAMES[feature].lower()} ({value:g}) is {word} {threshold:.1f}")
    if not reasons:
        return f"The model predicts {predicted}."
    return f"Predicted {predicted} because " + "; ".join(reasons) + "."


@app.post("/api/predict")
def predict():
    if STARTUP_ERROR:
        return model_unavailable()
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return error("Send a JSON object with the student's values (Content-Type: application/json).", 400)

    clean, problems = validate_payload(data)
    if problems:
        return error("Invalid input", 422, problems)

    row = pd.DataFrame([clean])[FEATURES]
    row["Participation_Level"] = row["Participation_Level"].map(PARTICIPATION_MAP)

    predicted = str(MODEL.predict(row)[0])
    probabilities = dict(zip(MODEL.classes_, MODEL.predict_proba(row)[0]))
    return jsonify(
        prediction=predicted,
        confidence=round(float(probabilities[predicted]), 4),
        probabilities={k: round(float(probabilities.get(k, 0.0)), 4) for k in LABELS},
        explanation=explain(row, predicted),
    )


# ---------------------------------------------------------------------------
# JSON errors for API calls + serving the website for local development
# (On Vercel the files in public/ are served by Vercel's CDN automatically.)
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_):
    return error("Not found", 404)


@app.errorhandler(405)
def method_not_allowed(_):
    return error("Method not allowed for this endpoint", 405)


@app.errorhandler(500)
def server_error(_):
    return error("Internal server error", 500)


@app.get("/")
def home():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.get("/<path:filename>")
def public_files(filename):
    if filename == "api/predict":  # exists, but only accepts POST
        return error("Method not allowed for this endpoint", 405)
    if filename.startswith("api/"):
        return error("Not found", 404)
    return send_from_directory(PUBLIC_DIR, filename)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
