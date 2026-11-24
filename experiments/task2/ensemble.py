"""Task 2 final: LightGBM + XGBoost + LogisticRegression, probability average, 5-fold (early stopping on an inner 10% split).
Run from repo root:  python experiments/task2/ensemble.py   -> task2/outputs/task2_pred.csv
OOF macro-F1 (5-fold, same protocol): lgb 0.7516 | xgb 0.7505 | logreg 0.7522 | average of the three 0.7542."""
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, warnings
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
D = Path("task2") if Path("task2/train.csv").exists() else Path(".")
tr, te = pd.read_csv(D / "train.csv"), pd.read_csv(D / "test.csv")
y = tr.segment.astype(int).values
cat = [c for c in tr.columns if c not in ("player_id", "segment") and not pd.api.types.is_numeric_dtype(tr[c])]
num = [c for c in tr.columns if c not in ["player_id", "segment", *cat] and not c.startswith("random_metric")]

def trees(d):                       # category codes from TRAIN only
    X = d[num].copy()
    for c in cat:
        X[c] = pd.Categorical(d[c], categories=sorted(tr[c].dropna().unique())).codes.astype(float)
        X.loc[d[c].isna(), c] = np.nan
    return X.replace(-1.0, np.nan) if False else X
def linear(d):
    return pd.concat([d[num], pd.get_dummies(d[cat], dtype=float)], axis=1)
Xt_tr, Xt_te = trees(tr), trees(te)
Xl_tr = linear(tr); Xl_te = linear(te).reindex(columns=Xl_tr.columns, fill_value=0.0)

oof = {k: np.zeros((len(y), 4)) for k in ("lgb", "xgb", "lr")}
pt = {k: np.zeros((len(te), 4)) for k in oof}
for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(Xt_tr, y)):
    inner = np.random.RandomState(f).rand(len(a)) < 0.1
    ai, vi = a[~inner], a[inner]
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.04, num_leaves=15, min_child_samples=40, subsample=.8, subsample_freq=1,
                           colsample_bytree=.5, reg_lambda=10, random_state=f, verbose=-1)
    m.fit(Xt_tr.iloc[ai], y[ai], eval_set=[(Xt_tr.iloc[vi], y[vi])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof["lgb"][b], pt["lgb"] = m.predict_proba(Xt_tr.iloc[b]), pt["lgb"] + m.predict_proba(Xt_te) / 5
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=0.04, max_depth=5, subsample=.8, colsample_bytree=.5, min_child_weight=5,
                          reg_lambda=5, tree_method="hist", early_stopping_rounds=100, random_state=f, n_jobs=-1)
    m.fit(Xt_tr.iloc[ai], y[ai], eval_set=[(Xt_tr.iloc[vi], y[vi])], verbose=False)
    oof["xgb"][b], pt["xgb"] = m.predict_proba(Xt_tr.iloc[b]), pt["xgb"] + m.predict_proba(Xt_te) / 5
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    A = sc.fit_transform(imp.fit_transform(Xl_tr.iloc[a]))
    m = LogisticRegression(C=0.3, max_iter=500).fit(A, y[a])
    tf = lambda X: sc.transform(imp.transform(X))
    oof["lr"][b], pt["lr"] = m.predict_proba(tf(Xl_tr.iloc[b])), pt["lr"] + m.predict_proba(tf(Xl_te)) / 5
    print("fold", f + 1, "done", flush=True)
for k, v in oof.items(): print(k, round(f1_score(y, v.argmax(1), average="macro"), 4))
avg_oof, avg_te = np.mean(list(oof.values()), 0), np.mean(list(pt.values()), 0)
print("ensemble OOF macro-F1:", round(f1_score(y, avg_oof.argmax(1), average="macro"), 4))
pred = avg_te.argmax(1)
print("test class share:", np.bincount(pred, minlength=4) / len(pred), "| train:", np.bincount(y) / len(y))
out = D / "outputs"; out.mkdir(exist_ok=True)
pd.DataFrame({"id": [f"ANS{i + 1:05d}" for i in range(len(pred))], "pred": pred}).to_csv(out / "task2_pred.csv", index=False)
