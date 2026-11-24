"""Task 3: OOF-weighted blend of the notebook models with more seeds.
Components (same 5-fold split as task3.ipynb): L1 LightGBM bagged over 5 seeds, the notebook's
quantile mixture, and a smaller regularised L1 LightGBM (3 seeds). Blend weights are searched on
out-of-fold predictions only; test predictions are the mean of the 5 fold models.
Run: python experiments/task3/exp_blend.py   (reuses cells 2,3,5 of task3.ipynb)"""
import itertools, json, os, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(Path(__file__).resolve().parents[2])
nb = json.load(open("task3/task3.ipynb", encoding="utf-8"))
g = {}
for i in (2, 3, 5):
    exec("".join(nb["cells"][i]["source"]), g)
np, pd, lgb, nmae, KFold = g["np"], g["pd"], g["lgb"], g["nmae"], g["KFold"]
train, test, ID, TARGET = g["train"], g["test"], g["ID"], g["TARGET"]
X = g["add_features"](train.drop(columns=[ID, TARGET]))
Xt = g["add_features"](test.drop(columns=[ID]))[X.columns]
y = train[TARGET].values

def small(seed):
    return lgb.LGBMRegressor(objective="l1", n_estimators=3000, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5, random_state=seed, verbose=-1)

names = ["l1_5seed", "qmix", "small_3seed"]
oof = {n: np.zeros(len(y)) for n in names}
tst = {n: np.zeros(len(Xt)) for n in names}
kf = KFold(5, shuffle=True, random_state=42)
for k, (tr, va) in enumerate(kf.split(X)):
    t0 = time.time()
    Xtr, Xva, ytr, yva = X.iloc[tr], X.iloc[va], y[tr], y[va]
    es = [lgb.early_stopping(100, verbose=False)]
    for name, makers in (("l1_5seed", [lambda s: g["make_l1"](3000).set_params(random_state=s)] * 1),
                         ("small_3seed", [small])):
        seeds = [42, 43, 44, 45, 46] if name == "l1_5seed" else [42, 43, 44]
        pv, pt = [], []
        for s in seeds:
            m = makers[0](s).fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1", callbacks=es)
            pv.append(m.predict(Xva)); pt.append(m.predict(Xt))
        oof[name][va] = np.clip(np.mean(pv, 0), 0, None)
        tst[name] += np.clip(np.mean(pt, 0), 0, None) / 5
    qm = g["fit_qmix"](Xtr, ytr)[0]
    oof["qmix"][va] = qm(Xva)
    tst["qmix"] += qm(Xt) / 5
    print(f"fold {k}: " + " | ".join(f"{n} {nmae(yva, oof[n][va]):.4f}" for n in names) + f" ({time.time() - t0:.0f}s)", flush=True)

for n in names:
    print(f"{n:12s} OOF NMAE {nmae(y, oof[n]):.4f}")
best = (None, 9.0)
grid = np.round(np.arange(0, 1.01, 0.05), 2)
for a, b in itertools.product(grid, grid):
    if a + b > 1:
        continue
    w = (a, b, round(1 - a - b, 2))
    s = nmae(y, np.clip(sum(wi * oof[n] for wi, n in zip(w, names)), 0, 300000))
    if s < best[1]:
        best = (w, s)
w = best[0]
print(f"notebook blend (L1 3 seeds + qmix): 0.2722 | old 50/50 of these: {nmae(y, 0.5 * oof['l1_5seed'] + 0.5 * oof['qmix']):.4f}")
print(f"best weights {dict(zip(names, w))} -> OOF NMAE {best[1]:.4f}")
pred = np.clip(sum(wi * tst[n] for wi, n in zip(w, names)), 0, 300000)
print(f"test pred: mean {pred.mean():.1f} | median {np.median(pred):.1f} | max {pred.max():.1f} | zeros {(pred == 0).mean():.3f}")
os.makedirs("task3/outputs", exist_ok=True)
pd.DataFrame({"id": [f"ANS{i + 1:05d}" for i in range(len(pred))], "pred": pred}).to_csv("task3/outputs/task3_pred_blend.csv", index=False)
print("saved task3/outputs/task3_pred_blend.csv")
