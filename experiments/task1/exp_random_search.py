"""Task 1: random hyperparameter search for LightGBM (own loop, no AutoML library).
Stage 1: sample configs, score each by 3-fold out-of-fold macro F2 (best threshold), until TIME_BUDGET.
Stage 2: refit the best config with the notebook's 5-fold split and 3 seeds; report OOF F2 and save test preds.
Only the labeled train rows are scored; test is used only for the final prediction.
Run: python experiments/task1/exp_random_search.py [time_budget_seconds]"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(Path(__file__).resolve().parents[2] / "task1")
TIME_BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1800
N_JOBS = int(os.environ.get("N_JOBS", "-1"))

train = pd.read_csv("train.csv"); test = pd.read_csv("test.csv")
base_cols = [c for c in test.columns if c != "player_id"]

def add_features(df):                       # same as task1.ipynb cell 3
    df = df.copy()
    df["friends_per_day"] = df["friend_network_size"] / (df["account_age_days"] + 1.0)
    df["reports_per_day"] = df["reports_received"] / (df["account_age_days"] + 1.0)
    df["reports_per_level"] = df["reports_received"] / (df["level"] + 1.0)
    df["level_per_day"] = df["level"] / (df["account_age_days"] + 1.0)
    df["aim_to_gamesense_ratio"] = (df["headshot_percentage"] + df["accuracy_score"]) / (df["game_sense_score"] + df["map_knowledge"] + 1e-5)
    df["kd_to_winrate_ratio"] = df["kill_death_ratio"] / (df["win_rate"] + 1.0)
    df["missing_count"] = df[base_cols].isnull().sum(axis=1)
    return df

lab = train.dropna(subset=["is_cheater"]).reset_index(drop=True)
F = [c for c in add_features(lab.head()).columns if c not in ("player_id", "is_cheater")]
X, y = add_features(lab)[F], lab["is_cheater"].astype(int).values
XT = add_features(test)[F]

def best_f2(yt, p):
    ts = np.arange(0.10, 0.90, 0.01)
    s = [fbeta_score(yt, (p >= t).astype(int), beta=2, average="macro") for t in ts]
    return float(ts[int(np.argmax(s))]), float(max(s))

def sample(rng):
    return dict(objective="binary", n_estimators=5000, verbose=-1,
                learning_rate=float(np.exp(rng.uniform(np.log(0.01), np.log(0.06)))),
                num_leaves=int(rng.choice([7, 15, 31, 63, 127, 255])),
                min_child_samples=int(rng.choice([10, 20, 50, 100, 200, 400])),
                colsample_bytree=float(rng.uniform(0.3, 1.0)),
                subsample=float(rng.uniform(0.5, 1.0)), subsample_freq=1,
                reg_lambda=float(rng.choice([0, 0.1, 1, 5, 10, 30])),
                reg_alpha=float(rng.choice([0, 0.1, 1, 5])),
                min_split_gain=float(rng.choice([0, 0, 0.01, 0.1])),
                max_bin=int(rng.choice([63, 255, 511])),
                is_unbalance=bool(rng.choice([False, True])))

def cv(params, n_splits, seeds, want_test=False):
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    oof, tst = np.zeros(len(y)), np.zeros(len(XT))
    for tr, va in skf.split(X, y):
        pv, pt = [], []
        for s in seeds:
            m = lgb.LGBMClassifier(**params, random_state=s, n_jobs=N_JOBS)
            m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
            pv.append(m.predict_proba(X.iloc[va])[:, 1])
            if want_test:
                pt.append(m.predict_proba(XT)[:, 1])
        oof[va] = np.mean(pv, 0)
        if want_test:
            tst += np.mean(pt, 0) / n_splits
    return oof, tst

rng = np.random.default_rng(2026)
NB = dict(objective="binary", learning_rate=0.05, num_leaves=31, n_estimators=1000, verbose=-1)
t0 = time.time()
rows = []
nb_score = best_f2(y, cv(NB, 3, [42])[0])[1]
print(f"notebook params, 3-fold: F2 {nb_score:.5f}", flush=True)
i = 0
while time.time() - t0 < TIME_BUDGET:
    p = sample(rng); t1 = time.time()
    thr, s = best_f2(y, cv(p, 3, [42])[0])
    rows.append({**{k: v for k, v in p.items() if k not in ("objective", "n_estimators", "verbose", "subsample_freq")}, "F2": s, "thr": thr})
    i += 1
    print(f"[{i}] F2 {s:.5f} (best {max(r['F2'] for r in rows):.5f}) {time.time() - t1:.0f}s | " +
          json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if k not in ('F2', 'thr')}), flush=True)
R = pd.DataFrame(rows).sort_values("F2", ascending=False)
os.makedirs("outputs", exist_ok=True)
R.to_csv("outputs/random_search.csv", index=False)
print("\ntop 5 configs (3-fold):\n", R.head(5).round(4).to_string(index=False))

best = {k: R.iloc[0][k] for k in R.columns if k not in ("F2", "thr")}
best = dict(objective="binary", n_estimators=5000, verbose=-1, subsample_freq=1,
            **{k: (int(v) if k in ("num_leaves", "min_child_samples", "max_bin") else bool(v) if k == "is_unbalance" else float(v)) for k, v in best.items()})
oof5, tst5 = cv(best, 5, [42, 43, 44], want_test=True)
thr, s5 = best_f2(y, oof5)
print(f"\nbest config, notebook 5-fold x 3 seeds: OOF macro F2 {s5:.5f} at threshold {thr:.2f} (notebook 0.77885)")
pd.DataFrame({"id": [f"ANS{i + 1:05d}" for i in range(len(XT))], "pred": (tst5 >= thr).astype(int), "proba": tst5}).to_csv(
    "outputs/task1_pred_search.csv", index=False)
np.save("outputs/task1_oof_search.npy", oof5)
print("saved outputs/task1_pred_search.csv | test positive share", round(float((tst5 >= thr).mean()), 4))
