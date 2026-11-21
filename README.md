# CPE342 Hackathon - Karena Player Intelligence

One folder per task, each with its `datacard.txt` and a self-contained notebook.

| folder | task | metric | notebook |
|---|---|---|---|
| `task1/` | Anti-cheat pre-filter (binary) | F2 macro | `task1.ipynb` |
| `task2/` | Player segment (4 classes) | F1 macro | `task2.ipynb` |
| `task3/` | Monthly spending (regression) | NMAE | `task3.ipynb` |
| `task4/` | Game title from screenshots | F1 macro | `task4.ipynb` (needs images + `torch`) |
| `task5/` | Account security (unsupervised anomaly) | F3 macro | `task5.ipynb` |

`sample_submission.csv` is the official template; `submission_final.csv` is the combined submission (`id, task1..task5`).

## Data
Every task folder follows the competition layout below. Data files are **not committed** (too large, see `.gitignore`) except
`task2/train.csv` / `task2/test.csv`: download them from the competition page and copy them into the matching folder.
Task 4 images: https://drive.google.com/file/d/1C8csylbBo3Oab24ODGKPVom3_SDUJ_Qw/view?usp=share_link (unzip to `task4/train/` and `task4/test/`).

```
CPE342-Hackathon/
├── task1/
│   ├── train.csv          # Training data with labels
│   ├── test.csv           # Test data for predictions
│   ├── datacard.txt       # Detailed feature descriptions
│   └── task1.ipynb
├── task2/
│   ├── train.csv
│   ├── test.csv
│   ├── datacard.txt
│   └── task2.ipynb
├── task3/
│   ├── train.csv
│   ├── test.csv
│   ├── datacard.txt
│   └── task3.ipynb
├── task4/
│   ├── train/             # Training images folder
│   ├── test/              # Test images folder
│   ├── train.csv          # Image labels and metadata
│   ├── test.csv           # Image metadata
│   ├── datacard.txt
│   └── task4.ipynb
├── task5/
│   ├── test.csv           # Test data (unsupervised)
│   ├── datacard.txt
│   └── task5.ipynb
├── sample_submission.csv  # Submission format template
└── submission_final.csv   # Combined submission (id, task1..task5)
```

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
