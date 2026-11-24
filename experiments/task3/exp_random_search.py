"""Task 3: random hyperparameter search for the L1 LightGBM (own loop, no AutoML library).
Stage 1: sample configs, score each by 3-fold OOF NMAE, until TIME_BUDGET.
Stage 2: best config with the notebook's 5-fold split x 3 seeds, plus the notebook's quantile mixture;
blend weight chosen on OOF; save test preds (mean of the 5 fold models).
Run: python experiments/task3/exp_random_search.py [time_budget_seconds]   (reuses cells 2,3,5 of task3.ipynb)"""
import json, os, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(Path(__file__).resolve().parents[2])
TIME_BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1800
N_JOBS = int(os.environ.get("N_JOBS", "-1"))
nb = json.load(open("task3/task3.ipynb", encoding="utf-8"))
g = {}
for i in (2, 3, 5):
    exec("".join(nb["cells"][i]["source"]), g)
np, pd, lgb, nmae, KFold = g["np"], g["pd"], g["lgb"], g["nmae"], g["KFold"]
train, test, ID, TARGET = g["train"], g["test"], g["ID"], g["TARGET"]
X = g["add_features"](train.drop(columns=[ID, TARGET]))
Xt = g["add_features"](test.drop(columns=[ID]))[X.columns]
y = train[TARGET].values

def sample(rng):
    return dict(objective="l1", n_estimators=6000, verbose=-1,
                learning_rate=float(np.exp(rng.uniform(np.log(0.015), np.log(0.08)))),
                num_leaves=int(rng.choice([15, 31, 63, 127, 255, 511])),
                min_child_samples=int(rng.choice([5, 10, 20, 40, 80, 160])),
                colsample_bytree=float(rng.uniform(0.4, 1.0)),
                subsample=float(rng.uniform(0.6, 1.0)), subsample_freq=1,
                reg_lambda=float(rng.choice([0, 0.1, 1, 5, 20])),
                reg_alpha=float(rng.choice([0, 0.1, 1, 5])),
                max_bin=int(rng.choice([63, 255, 511, 1023])))

def cv(params, n_splits, seeds, want_test=False):
    kf = KFold(n_splits, shuffle=True, random_state=42)
    oof, tst = np.zeros(len(y)), np.zeros(len(Xt))
    for tr, va in kf.split(X):
        pv, pt = [], []
        for s in seeds:
            m = lgb.LGBMRegressor(**params, random_state=s, n_jobs=N_JOBS)
            m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="l1",
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            pv.append(m.predict(X.iloc[va]))
            if want_test:
                pt.append(m.predict(Xt))
        oof[va] = np.clip(np.mean(pv, 0), 0, None)
        if want_test:
            tst += np.clip(np.mean(pt, 0), 0, None) / n_splits
    return oof, tst

rng = np.random.default_rng(2026)
t0 = time.time()
base = g["make_l1"](6000).get_params()
base = {k: base[k] for k in ("objective", "learning_rate", "num_leaves", "min_child_samples", "subsample", "subsample_freq", "colsample_bytree")}
base.update(n_estimators=6000, verbose=-1)
print(f"notebook L1 params, 3-fold: NMAE {nmae(y, cv(base, 3, [42])[0]):.4f}", flush=True)
rows, i = [], 0
while time.time() - t0 < TIME_BUDGET:
    p = sample(rng); t1 = time.time()
    s = nmae(y, cv(p, 3, [42])[0])
    rows.append({**{k: v for k, v in p.items() if k not in ("objective", "n_estimators", "verbose", "subsample_freq")}, "NMAE": s})
    i += 1
    print(f"[{i}] NMAE {s:.4f} (best {min(r['NMAE'] for r in rows):.4f}) {time.time() - t1:.0f}s | " +
          json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != 'NMAE'}), flush=True)
R = pd.DataFrame(rows).sort_values("NMAE")
os.makedirs("task3/outputs", exist_ok=True)
R.to_csv("task3/outputs/random_search.csv", index=False)
print("\ntop 5 configs (3-fold):\n", R.head(5).round(4).to_string(index=False))

best = dict(objective="l1", n_estimators=6000, verbose=-1, subsample_freq=1,
            **{k: (int(v) if k in ("num_leaves", "min_child_samples", "max_bin") else float(v))
               for k, v in R.iloc[0].items() if k != "NMAE"})
oof_l1, tst_l1 = cv(best, 5, [42, 43, 44], want_test=True)
kf = KFold(5, shuffle=True, random_state=42)
oof_qm, tst_qm = np.zeros(len(y)), np.zeros(len(Xt))
for tr, va in kf.split(X):
    qm = g["fit_qmix"](X.iloc[tr], y[tr])[0]
    oof_qm[va] = qm(X.iloc[va]); tst_qm += qm(Xt) / 5
print(f"\nbest L1 config 5-fold x 3 seeds: {nmae(y, oof_l1):.4f} | quantile mixture: {nmae(y, oof_qm):.4f}")
ws = np.round(np.arange(0, 1.01, 0.05), 2)
sc = [nmae(y, np.clip(w * oof_l1 + (1 - w) * oof_qm, 0, 300000)) for w in ws]
w = ws[int(np.argmin(sc))]
print(f"blend weight L1={w:.2f} -> OOF NMAE {min(sc):.4f} (notebook blend 0.2722)")
pred = np.clip(w * tst_l1 + (1 - w) * tst_qm, 0, 300000)
print(f"test pred: mean {pred.mean():.1f} | median {np.median(pred):.1f} | max {pred.max():.1f}")
pd.DataFrame({"id": [f"ANS{i + 1:05d}" for i in range(len(pred))], "pred": pred}).to_csv("task3/outputs/task3_pred_search.csv", index=False)
np.save("task3/outputs/task3_oof_search.npy", np.vstack([oof_l1, oof_qm]))
print("saved task3/outputs/task3_pred_search.csv")
