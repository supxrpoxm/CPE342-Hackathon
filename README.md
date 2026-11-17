# CPE342 Hackathon - Karena Player Intelligence

One folder per task, each with its `datacard.txt` and a self-contained notebook.

| folder | task | metric | notebook |
|---|---|---|---|
| `task1/` | Anti-cheat pre-filter (binary) | F2 macro | `task1.ipynb` |
| `task2/` | Player segment (4 classes) | F1 macro | `task2.ipynb` |
| `task3/` | Monthly spending (regression) | NMAE | `task3.ipynb` |
| `task4/` | Game title from screenshots | F1 macro | `task4.ipynb` (needs images + `torch`) |
| `task5/` | Account security (unsupervised anomaly) | F3 macro | `task5.ipynb` |

A task's folder appears once its owner has pushed it, so not every folder is present on every branch.
`sample_submission.csv` is the official template; `submission_final.csv` is the combined submission (`id, task1..task5`).

## Data
Competition data is not committed for large or per-person files (see `.gitignore`). Download `train.csv` / `test.csv` from the
competition page and copy them into the matching `taskN/` folder before running its notebook. If a task's data is already
tracked in the repo (e.g. `task2/`), nothing needs to be copied.

## Run
```bash
pip install numpy pandas scikit-learn lightgbm matplotlib joblib nbconvert
cd taskN && jupyter nbconvert --to notebook --execute --inplace taskN.ipynb
```
Each task may need extra packages (e.g. `catboost`, `torch`); check the first cell of its notebook.
Notebooks work from the repo root or from inside their own `taskN/` folder. Where supported, `QUICK=1` runs a small smoke test.
Generated outputs (`outputs/`, `*.joblib`) are git-ignored; re-create them by running the notebook.

## Working together
- Work on your own task folder and avoid editing another task's files.
- `submission_final.csv` is shared by everyone, so pull before changing it and only touch your own task's column.
- Do not commit data, model files or secrets.

## Rules
Models are trained from scratch (pre-trained weights only where the task allows it, e.g. images), no AutoML. Every statistic
(clip bounds, imputer, scaler, thresholds) should be computed from the train set only.
Task 5 has no train set and no labels, so its unsupervised detectors necessarily fit on the features of `test.csv`; this was
flagged to confirm with the TA, and the flag rate is chosen by synthetic validation, not from leaderboard scores.
