# Predicting Student Academic Performance Using Machine Learning

A Flask + scikit-learn web app. A real Python backend trains a **Decision Tree** on a CSV of 500 synthetic students and predicts whether a new student is **Low**, **Average** or **High**. The data is randomly generated; no real personal information is used.

## Project structure

```
api/index.py           Flask backend (the `app` object Vercel runs)
public/                Website: index.html, style.css, script.js
data/students.csv      Dataset (500 rows)
scripts/generate_dataset.py   Script that created the CSV (optional)
tests/test_api.py      Automated tests (optional)
requirements.txt       Python packages
vercel.json            Vercel routing for /api/*
```

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Backend status |
| GET | `/api/metrics` | Dataset size, class counts, test accuracy, confusion matrix, labels |
| GET | `/api/students?limit=10` | First rows of the CSV (preview table) |
| POST | `/api/predict` | Predict a student's class |

Example `POST /api/predict` body:

```json
{
  "Attendance_Percentage": 85,
  "Study_Hours_Per_Day": 4,
  "Previous_Marks": 72,
  "Assignment_Completion_Percentage": 88,
  "Sleep_Hours": 7,
  "Participation_Level": "High"
}
```

Errors are returned as JSON: `400` (not valid JSON), `422` (missing or out-of-range values, with a `details` list), `404`, `405`.

## Run locally (Windows)

1. Install Python 3.10+ from https://www.python.org/downloads/ (tick **Add python.exe to PATH**).
2. Extract the ZIP, open the folder, click the address bar, type `cmd`, press Enter.
3. Run these commands one by one:

```
python --version
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python api\index.py
```

(If `python` is not recognised, use `py` instead.)

4. Open http://127.0.0.1:5000 in your browser.
5. Test the endpoints (open a **second** Command Prompt for the POST test):
   - http://127.0.0.1:5000/api/health
   - http://127.0.0.1:5000/api/metrics
   - Prediction (PowerShell):

```
powershell -Command "Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/predict -ContentType 'application/json' -Body '{\"Attendance_Percentage\":85,\"Study_Hours_Per_Day\":4,\"Previous_Marks\":72,\"Assignment_Completion_Percentage\":88,\"Sleep_Hours\":7,\"Participation_Level\":\"High\"}'"
```

   You can also just use the form on the website.

6. Optional tests: `python -m unittest discover tests -v`
7. Stop the server with `Ctrl + C`.

## Deploy on Vercel (via GitHub website)

1. Create a GitHub repository on github.com and upload the **contents** of the project folder (so `api`, `public`, `data`, `requirements.txt`, `vercel.json` are at the top level). Do not upload `venv`.
2. On vercel.com choose **Add New → Project**, import the repository and click **Deploy**. Leave all settings at their defaults.
3. Open the live URL and check `/api/health`.

## How it works

1. `pandas` loads and validates `data/students.csv`.
2. Participation_Level is mapped Low/Medium/High → 1/2/3 (a fixed mapping, nothing is learned from the data, so no leakage).
3. `train_test_split` (80/20, `random_state=42`, stratified) happens **before** training.
4. `DecisionTreeClassifier(max_depth=4)` is trained on the training part only.
5. Accuracy and the confusion matrix are computed from predictions on the held-out test part. The same trained model answers `/api/predict`.
6. Training happens once when the server starts and is kept in memory (no retraining per request).

## Limitations

- Synthetic data: results demonstrate the method, not real student behaviour.
- Small dataset (500 rows) and only 6 features.
- A single Decision Tree is sensitive to small changes in the data.
- Do not use predictions to make decisions about real students.
