"""Task 5: synthetic check of combined detectors at the final 12 % flag rate (and the 5 % reference).
Same injected scenarios as task5.ipynb section 5.3 (two injection seeds each); no test labels or leaderboard.
New candidates:
  C1  max over 3 groups (IF+kNN) and a 4th "global" group (IF+kNN on all non-flag features)
  C2  rank average of '3 groups: max (IF+kNN)' and 'global ensemble (IF+PCA+kNN)'
  C3  max of '3 groups: max (IF+kNN)' and 'global ensemble (IF+PCA+kNN)'
Run: python experiments/task5/exp_combo.py"""
import json, os, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(Path(__file__).resolve().parents[2])
nb = json.load(open("task5/task5.ipynb", encoding="utf-8"))
g = {}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code" and i <= 18:
        exec("".join(c["source"]), g)
src20 = "".join(nb["cells"][20]["source"])
exec(src20.split("SCENARIOS = {")[0], g)          # inject() and f3_macro() only
np, pd = g["np"], g["pd"]
SCEN = {"threats / flags informative": dict(strength=1.0, flag_signal=True),
        "threats / flags uninformative": dict(strength=1.0, flag_signal=False),
        "threats / weak (60 %)": dict(strength=0.6, flag_signal=False),
        "generic unseen type": dict(kind="generic")}
CUR, GLOB = "3 groups: max (IF+kNN)", "global ensemble (IF+PCA+kNN)"

def candidates(S):
    V = g["variant_scores"](S)
    G = list(g["GROUP_COLS"])
    four = np.column_stack([np.mean([S[(k, False, gg)] for k in ("if", "knn")], 0) for gg in G + ["global"]])
    return {CUR: V[CUR], GLOB: V[GLOB],
            "C1 4 groups incl. global: max (IF+kNN)": four.max(1),
            "C2 rank avg (current, global ens.)": g["pct"](g["pct"](V[CUR]) + g["pct"](V[GLOB])),
            "C3 max (current, global ens.)": np.maximum(g["pct"](V[CUR]), g["pct"](V[GLOB]))}

rows, t0 = [], time.time()
for scen, kw in SCEN.items():
    for k in range(2):
        d_syn, y = g["inject"](g["df"], 100 + k, **kw)
        S, _ = g["detector_scores"](g["prepare"](d_syn), 42 + k, 150)
        for name, sc in candidates(S).items():
            for rate in (0.05, 0.12):
                rows.append({"scenario": scen, "variant": name, "rate": rate, "F3": g["f3_macro"](y, g["top_rate_flags"](sc, rate))})
    print(f"{scen} done ({time.time() - t0:.0f}s)", flush=True)
R = pd.DataFrame(rows)
for rate in (0.12, 0.05):
    t = R[R.rate == rate].pivot_table(index="variant", columns="scenario", values="F3")[list(SCEN)]
    t["avg"], t["worst"] = t.mean(1), t[list(SCEN)].min(1)
    print(f"\nF3 macro at flag rate {rate:.0%} (mean of 2 injections)\n", t.sort_values("avg", ascending=False).round(3).to_string())
R.to_csv("task5/outputs/exp_combo.csv", index=False)
