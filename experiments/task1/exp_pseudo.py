"""Task 1: stronger LightGBM and pseudo-labelling of the unlabeled train rows, vs the notebook.
A  notebook setup: notebook features + params (lr 0.05, 31 leaves), 1 seed
B  tuned: lr 0.02, 63 leaves, feature/bagging fraction, L2, 3 seeds
C  B + pseudo-labels: inside each fold, the fold model labels the train rows that have no
   is_cheater; confident ones (p < LO or p > HI) are added with hard labels, then B is refit.
All scored by out-of-fold macro F2 at the best OOF threshold, same 5-fold split as the notebook.
Run: python experiments/task1/exp_pseudo.py"""
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(Path(__file__).resolve().parents[2] / "task1")

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
unl = train[train["is_cheater"].isna()].reset_index(drop=True)
F = [c for c in add_features(lab.head()).columns if c not in ("player_id", "is_cheater")]
X, y = add_features(lab)[F], lab["is_cheater"].astype(int).values
XU = add_features(unl)[F]
XT = add_features(test)[F]
print(f"labeled {len(X)} | unlabeled {len(XU)} | test {len(XT)} | positive share {y.mean():.3f}")

P_NB = dict(objective="binary", learning_rate=0.05, num_leaves=31, n_estimators=1000, verbose=-1)
P_TUNED = dict(objective="binary", learning_rate=0.02, num_leaves=63, n_estimators=5000, min_child_samples=50,
               colsample_bytree=0.7, subsample=0.8, subsample_freq=1, reg_lambda=1.0, verbose=-1)
LO, HI = 0.10, 0.90

def fit(params, Xtr, ytr, Xva, yva, seed):
    m = lgb.LGBMClassifier(**params, random_state=seed)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100 if params is P_TUNED else 50, verbose=False)])
    return m

def best_f2(yt, p):
    ts = np.arange(0.10, 0.90, 0.01)
    s = [fbeta_score(yt, (p >= t).astype(int), beta=2, average="macro") for t in ts]
    return ts[int(np.argmax(s))], max(s)

skf = StratifiedKFold(5, shuffle=True, random_state=42)
names = ["A_notebook", "B_tuned", "C_tuned_pseudo"]
oof = {n: np.zeros(len(y)) for n in names}
tst = {n: np.zeros(len(XT)) for n in names}
n_pseudo = []
for k, (tr, va) in enumerate(skf.split(X, y)):
    t0 = time.time()
    Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y[tr], y[va]
    m = fit(P_NB, Xtr, ytr, Xva, yva, 42)
    oof["A_notebook"][va] = m.predict_proba(Xva)[:, 1]; tst["A_notebook"] += m.predict_proba(XT)[:, 1] / 5
    pv, pt, pu = [], [], []
    for s in (42, 43, 44):
        m = fit(P_TUNED, Xtr, ytr, Xva, yva, s)
        pv.append(m.predict_proba(Xva)[:, 1]); pt.append(m.predict_proba(XT)[:, 1]); pu.append(m.predict_proba(XU)[:, 1])
    oof["B_tuned"][va] = np.mean(pv, 0); tst["B_tuned"] += np.mean(pt, 0) / 5
    pu = np.mean(pu, 0)
    keep = (pu < LO) | (pu > HI)
    n_pseudo.append(int(keep.sum()))
    Xaug = pd.concat([Xtr, XU[keep]], ignore_index=True)
    yaug = np.concatenate([ytr, (pu[keep] > 0.5).astype(int)])
    pv, pt = [], []
    for s in (42, 43, 44):
        m = fit(P_TUNED, Xaug, yaug, Xva, yva, s)
        pv.append(m.predict_proba(Xva)[:, 1]); pt.append(m.predict_proba(XT)[:, 1])
    oof["C_tuned_pseudo"][va] = np.mean(pv, 0); tst["C_tuned_pseudo"] += np.mean(pt, 0) / 5
    print(f"fold {k}: " + " | ".join(f"{n} F2 {best_f2(yva, oof[n][va])[1]:.4f}" for n in names)
          + f" | pseudo rows {keep.sum()} ({time.time() - t0:.0f}s)", flush=True)

res = {}
for n in names:
    t, s = best_f2(y, oof[n])
    res[n] = (t, s)
    print(f"{n:15s} OOF macro F2 {s:.5f} at threshold {t:.2f} | test positive share {(tst[n] >= t).mean():.3f}")
os.makedirs("outputs", exist_ok=True)
for n in names[1:]:
    t = res[n][0]
    pd.DataFrame({"id": [f"ANS{i + 1:05d}" for i in range(len(XT))], "pred": (tst[n] >= t).astype(int), "proba": tst[n]}).to_csv(
        f"outputs/task1_pred_{n}.csv", index=False)
print("saved outputs/task1_pred_B_tuned.csv and outputs/task1_pred_C_tuned_pseudo.csv")
