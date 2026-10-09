"""Generate the synthetic dataset data/students.csv (500 fake students).

Run from the project root:  python scripts/generate_dataset.py
All values are randomly generated - no real personal information.
A fixed random seed makes the output identical every time.
"""
from pathlib import Path

import numpy as np
import pandas as pd

N = 500
rng = np.random.default_rng(42)

# A hidden "ability" value makes the columns realistically correlated.
ability = rng.normal(0, 1, N)

previous_marks = np.clip(62 + 14 * ability + rng.normal(0, 6, N), 25, 100)
attendance = np.clip(76 + 8 * ability + rng.normal(0, 9, N), 40, 100)
study_hours = np.clip(3.4 + 0.9 * ability + rng.normal(0, 1.2, N), 0.3, 9)
assignment = np.clip(72 + 8 * ability + rng.normal(0, 10, N), 30, 100)
sleep = np.clip(rng.normal(6.8, 1.1, N), 4, 9.5)

p = ability + rng.normal(0, 0.8, N)
participation = np.where(p < -0.5, "Low", np.where(p > 0.6, "High", "Medium"))
part_num = pd.Series(participation).map({"Low": 1, "Medium": 2, "High": 3}).to_numpy()

# Overall score = weighted mix of the features + random noise (so it is not perfectly predictable).
score = (
    0.35 * previous_marks
    + 0.20 * attendance
    + 0.20 * assignment
    + 2.5 * study_hours
    - 1.0 * np.abs(sleep - 7.5)
    + 3.0 * part_num
    + rng.normal(0, 2.0, N)
)
low_cut, high_cut = np.quantile(score, [0.28, 0.70])
performance = np.where(score < low_cut, "Low", np.where(score >= high_cut, "High", "Average"))

df = pd.DataFrame(
    {
        "Student_ID": [f"STU{i:03d}" for i in range(1, N + 1)],
        "Attendance_Percentage": attendance.round(1),
        "Study_Hours_Per_Day": study_hours.round(1),
        "Previous_Marks": previous_marks.round(1),
        "Assignment_Completion_Percentage": assignment.round(1),
        "Sleep_Hours": sleep.round(1),
        "Participation_Level": participation,
        "Performance": performance,
    }
)

out = Path(__file__).resolve().parent.parent / "data" / "students.csv"
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
print(df["Performance"].value_counts())
