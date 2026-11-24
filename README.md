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
├── experiments/           # trial scripts, not part of the final submission
├── requirements.txt
├── sample_submission.csv  # Submission format template
└── submission_final.csv   # Combined submission (id, task1..task5)
```

## Run
```bash
pip install -r requirements.txt
cd taskN && jupyter nbconvert --to notebook --execute --inplace taskN.ipynb
```
Notebooks work from the repo root or from inside their own `taskN/` folder. Where supported, `QUICK=1` runs a small smoke test.
Task 4 needs a GPU; the notebook also runs on Colab and reads the images from Google Drive.
Generated outputs (`outputs/`, `*.joblib`) are git-ignored; re-create them by running the notebook.

## Final submission
`submission_final.csv` is the file selected on Kaggle (leaderboard **0.78760**). Final model per task:

| task | final model (in `taskN/taskN.ipynb`) |
|---|---|
| 1 | 5-fold LightGBM, decision threshold tuned for out-of-fold macro F2 |
| 2 | LightGBM + XGBoost + LogisticRegression; blend weights and per-class weights tuned on out-of-fold predictions (section 5.4) |
| 3 | seed-bagged L1 LightGBM + quantile-mixture blend |
| 4 | 5-fold ConvNeXt-Small + EfficientNetV2-S ensemble (ImageNet pre-trained, allowed for this task) |
| 5 | Isolation Forest + kNN detectors per threat group, flag rate 12% chosen by synthetic-anomaly validation |

## Experiments
`experiments/taskN/` holds scripts we used to try ideas. They are **not** part of the final submission; results are
in the report. Run each from the repo root, e.g. `python experiments/task3/exp_blend.py` (each file's docstring says what it tests).

| file | what it tests | outcome |
|---|---|---|
| `task1/exp_pseudo.py` | tuned LightGBM, pseudo-labels for the unlabeled **train** rows | small CV gain, worse on the leaderboard; not used |
| `task1/exp_random_search.py` | random hyper-parameter search (own loop, no AutoML) | not used |
| `task2/ensemble.py`, `task2/exp_classweights.py` | ensemble + blend/class weight tuning with nested CV | moved into `task2.ipynb` section 5.4 (final) |
| `task2/exp_ablation.py` | feature ablations on an earlier task 2 notebook (not in the repo) | kept for reference |
| `task3/exp_extra_models.py`, `exp_features.py`, `exp_blend.py`, `exp_random_search.py` | extra models, features, blends, hyper-parameter search | small CV gains, worse on the leaderboard; not used |
| `task4/task4_fast.ipynb` | faster ConvNeXt-Tiny 160x320 pipeline for Colab | leaderboard lower than the final model; not used |
| `task5/exp_flag_rate.py`, `exp_combo.py` | wider flag-rate grid and detector combinations on synthetic anomalies | led to the 12% flag rate in `task5.ipynb` |

## Working together
- Work on your own task folder and avoid editing another task's files.
- `submission_final.csv` is shared by everyone, so pull before changing it and only touch your own task's column.
- Do not commit data, model files or secrets.

## Rules
Models are trained from scratch (pre-trained weights only where the task allows it, e.g. images), no AutoML. Every statistic
(clip bounds, imputer, scaler, thresholds) should be computed from the train set only.
Task 5 has no train set and no labels, so its unsupervised detectors necessarily fit on the features of `test.csv` (no labels or
pseudo-labels are used), and the flag rate is chosen by synthetic validation, not from leaderboard scores.
