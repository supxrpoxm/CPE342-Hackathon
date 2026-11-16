# CPE342 Hackathon - Karena Player Intelligence (branch `Fuse`)

One folder per task, each with its `datacard.txt` and a self-contained notebook. `train.csv` / `test.csv` are **not tracked** for tasks 1, 3, 5 (git-ignored; copy them from the competition download into `taskN/`); task 2's data is already tracked.

## Branches
- `Fuse` - base: this README, `.gitignore`, `sample_submission.csv` and **task 5** (the part owned by the author).
- `fuse-task1` ... `fuse-task4` - experimental attempts at tasks 1-4, branched from `Fuse` and kept as history (they can be extended later). They are not meant to be merged into the shared `task1`-`task4` branches.

| folder | task | metric | notebook |
|---|---|---|---|
| `task1/` | Anti-cheat pre-filter (binary) | F2 macro | `task1.ipynb` |
| `task2/` | Player segment (4 classes) | F1 macro | `task2.ipynb` |
| `task3/` | Monthly spending (regression) | NMAE | `task3.ipynb` |
| `task4/` | Game title from screenshots | F1 macro | `task4.ipynb` (needs images + `torch`, see `task4/README.md`) |
| `task5/` | Account security (unsupervised anomaly) | F3 macro | `task5.ipynb` |

`submission_final.csv` is the combined submission (`id, task1..task5`), `sample_submission.csv` is the official template.

## Run
```bash
pip install numpy pandas scikit-learn lightgbm matplotlib joblib nbconvert
cd task1 && jupyter nbconvert --to notebook --execute --inplace task1.ipynb
```
Notebooks work from the repo root or from inside their own `taskN/` folder. `QUICK=1` runs a small smoke test.
Outputs (`outputs/`: predictions, models) are generated and git-ignored.

## Rules followed
Models are trained from scratch (pre-trained only for task 4), no AutoML; every statistic (clip bounds, imputer, scaler, thresholds)
is computed from the train set only. Task 5 has no train set, so its unsupervised detectors necessarily fit on `test.csv` features (no labels) - check this with the TA.
