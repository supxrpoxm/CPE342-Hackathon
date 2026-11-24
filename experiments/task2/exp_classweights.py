"""Task 2: on top of ensemble.py (LGBM + XGB + LR), tune (1) the model blend weights and (2) per-class
probability multipliers for macro F1 on out-of-fold predictions. The gain is checked with nested CV
(tune on 4/5 of the OOF rows, score on the held-out 1/5), so it is not an in-sample number.
Run: python experiments/task2/exp_classweights.py"""
import itertools, os, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(Path(__file__).resolve().parents[2])
g = {"__name__": "ens"}
exec(open("experiments/task2/ensemble.py", encoding="utf-8").read(), g)
np, pd, f1_score, y, oof, pt = g["np"], g["pd"], g["f1_score"], g["y"], g["oof"], g["pt"]
from sklearn.model_selection import StratifiedKFold
K = ["lgb", "xgb", "lr"]
f1 = lambda yy, P: f1_score(yy, P.argmax(1), average="macro")

def fit_blend(idx):
    best = (None, -1)
    for a, b in itertools.product(np.arange(0, 1.01, 0.1), repeat=2):
        if a + b > 1: continue
        w = np.array([a, b, 1 - a - b])
        s = f1(y[idx], sum(wi * oof[k][idx] for wi, k in zip(w, K)))
        if s > best[1]: best = (w, s)
    return best[0]

def fit_cw(P, yy):
    w = np.ones(4)
    for _ in range(3):
        for c in range(4):
            cand = np.arange(0.6, 1.61, 0.02)
            sc = []
            for v in cand:
                ww = w.copy(); ww[c] = v; sc.append(f1(yy, P * ww))
            w[c] = cand[int(np.argmax(sc))]
    return w

blend = lambda w, D: sum(wi * D[k] for wi, k in zip(w, K))
base_avg = np.mean([oof[k] for k in K], 0)
print(f"plain average OOF F1 {f1(y, base_avg):.4f}")
outer = StratifiedKFold(5, shuffle=True, random_state=7)
held_base, held_new = [], []
for a, b in outer.split(base_avg, y):
    wb = fit_blend(a); P = blend(wb, oof)
    wc = fit_cw(P[a], y[a])
    held_base.append(f1(y[b], base_avg[b])); held_new.append(f1(y[b], P[b] * wc))
print(f"nested CV: plain average {np.mean(held_base):.4f} | tuned blend + class weights {np.mean(held_new):.4f} "
      f"(gain {np.mean(held_new) - np.mean(held_base):+.4f}, per split {np.round(np.array(held_new) - np.array(held_base), 4).tolist()})")
wb = fit_blend(np.arange(len(y))); P = blend(wb, oof); wc = fit_cw(P, y)
print(f"final: blend weights {dict(zip(K, np.round(wb, 2)))} | class weights {np.round(wc, 2).tolist()} | in-sample OOF F1 {f1(y, P * wc):.4f}")
T = blend(wb, pt) * wc
pred = T.argmax(1)
print("test class share:", np.round(np.bincount(pred, minlength=4) / len(pred), 3), "| train:", np.round(np.bincount(y) / len(y), 3))
pd.DataFrame({"id": [f"ANS{i + 1:05d}" for i in range(len(pred))], "pred": pred}).to_csv("task2/outputs/task2_pred_tuned.csv", index=False)
np.savez("task2/outputs/task2_oof_ens.npz", **{f"oof_{k}": oof[k] for k in K}, **{f"pt_{k}": pt[k] for k in K})
print("saved task2/outputs/task2_pred_tuned.csv")
