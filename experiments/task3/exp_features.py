"""Task 3 experiment: extra "expected monthly spend" features vs the notebook features.
Same 5-fold split (seed 42) and same L1 LightGBM as task3.ipynb, one seed, so the
numbers are directly comparable with l1_base in exp_extra_models.py.
Run: python experiments/task3/exp_features.py   (reuses cells 2,3,5 of task3.ipynb)"""
import json, os, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
nb = json.load(open("task3/task3.ipynb", encoding="utf-8"))
g = {}
for i in (2, 3, 5):
    exec("".join(nb["cells"][i]["source"]), g)
np, pd, lgb, nmae, KFold = g["np"], g["pd"], g["lgb"], g["nmae"], g["KFold"]
train, ID, TARGET = g["train"], g["ID"], g["TARGET"]


def add_more(d):
    d = d.copy()
    months = d["account_age_days"] / 30 + 1
    d["exp_monthly_txn"] = d["avg_transaction_value"] * d["purchase_frequency"]
    d["exp_monthly_hist"] = d["avg_transaction_value"] * d["total_transactions"] / months
    d["hist_per_month"] = d["historical_spending"] / months
    d["prev_vs_hist_month"] = d["prev_month_spending"] / (d["hist_per_month"] + 1)
    d["txn_value_recency"] = d["avg_transaction_value"] / (d["days_since_last_purchase"] + 1)
    d["prev_x_freq"] = d["prev_month_spending"] * d["purchase_frequency"]
    d["prev_x_vip"] = d["prev_month_spending"] * (d["vip_status"].fillna(0) + 1)
    d["full_price_share"] = 1 - d["purchases_on_discount"] / (d["total_transactions"] + 1)
    seg_med = train.groupby("segment")["prev_month_spending"].median()   # train-only statistic
    d["prev_vs_segment"] = d["prev_month_spending"] / (d["segment"].map(seg_med) + 1)
    return d.replace([np.inf, -np.inf], np.nan)


X0 = g["add_features"](train.drop(columns=[ID, TARGET]))
X1 = add_more(X0)
y = train[TARGET].values
kf = KFold(5, shuffle=True, random_state=42)
oof = {"base": np.zeros(len(y)), "more": np.zeros(len(y))}
for k, (tr, va) in enumerate(kf.split(X0)):
    t = time.time()
    for name, X in (("base", X0), ("more", X1)):
        m = g["make_l1"](3000)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], eval_metric="l1",
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[name][va] = np.clip(m.predict(X.iloc[va]), 0, None)
    print(f"fold {k}: base {nmae(y[va], oof['base'][va]):.4f} | more {nmae(y[va], oof['more'][va]):.4f} ({time.time() - t:.0f}s)", flush=True)
print(f"CV base {nmae(y, oof['base']):.4f} | more {nmae(y, oof['more']):.4f}")
imp = pd.Series(m.feature_importances_, X1.columns).sort_values(ascending=False)
print("top features (last fold, 'more'):\n", imp.head(15).to_string())
os.makedirs("task3/outputs", exist_ok=True)
np.save("task3/outputs/oof_features_exp.npy", np.vstack([oof["base"], oof["more"]]))
