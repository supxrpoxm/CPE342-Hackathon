"""Task 5: re-run the synthetic flag-rate check of task5.ipynb (section 5.4) with a wider grid.
The notebook grid stopped at 12 %, where its own min-max-regret pick landed. Same injected
scenarios and prevalences; no test labels or leaderboard scores are used.
Run: python experiments/task5/exp_flag_rate.py"""
import json, os, sys, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(Path(__file__).resolve().parents[2])
nb = json.load(open("task5/task5.ipynb", encoding="utf-8"))
g = {}
t0 = time.time()
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code" or i > 22:
        continue
    src = "".join(c["source"])
    if i == 22:
        src = src.replace("RATES = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]",
                          "RATES = [0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25]")
    exec(src, g)
    print(f"[cell {i} done, {time.time() - t0:.0f}s]", flush=True)
g["R"].to_csv("task5/outputs/flag_rate_wide.csv", index=False)
