"""Trial-level preprocessing for the overcaution project.

Outputs (results/):
  trials.csv   one row per trial kept for modelling
  subjects.csv one row per included subject (ITI per load, counts, exclusion flags)
"""
import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results")
os.makedirs(OUT, exist_ok=True)

RT_MIN, RT_MAX = 0.2, 3.0          # s, likelihood window (also used for simulated data)
REWARD = 20.0
BLOCK_SEC = 60.0


def main():
    t = pd.read_csv(os.path.join(BASE, "data", "Behavioral_Scores_trials.csv"))
    t = t.sort_values(["Subject", "TaskBlock", "RunningScore"], kind="stable")
    # keep file order inside block (SubTrial is the within-block counter)
    t = t.sort_values(["Subject", "TaskBlock", "SubTrial"], kind="stable").reset_index(drop=True)
    t["load"] = (t.Load_Condition == "Load").astype(int)
    t["rt"] = t.RT / 1000.0
    t["acc"] = t.ACC.astype(int)
    t["pos"] = t.groupby(["Subject", "TaskBlock"]).cumcount()
    t["prev_acc"] = t.groupby(["Subject", "TaskBlock"]).acc.shift()

    # per-trial overhead (feedback + ITI [+ letter display under load]) from the fixed 60 s budget
    b = t.groupby(["Subject", "TaskBlock", "load"]).agg(n=("rt", "size"), sumrt=("rt", "sum")).reset_index()
    b["iti"] = (BLOCK_SEC - b.sumrt) / b.n
    iti = b.groupby(["Subject", "load"]).iti.median().unstack()
    iti.columns = ["iti_noload", "iti_load"]

    s = t.groupby("Subject").agg(n=("rt", "size"), acc=("acc", "mean"),
                                 in_win=("rt", lambda x: ((x >= RT_MIN) & (x <= RT_MAX)).mean()))
    s = s.join(iti)
    s["include"] = (s.acc >= 0.70) & (s.in_win >= 0.85) & s.iti_load.between(0.3, 1.5) & s.iti_noload.between(0.3, 1.5)
    s.to_csv(os.path.join(OUT, "subjects.csv"))
    print(f"subjects: {len(s)} total, {s.include.sum()} included")
    print(s[s.include][["iti_noload", "iti_load"]].describe().round(3))

    keep = t[t.Subject.isin(s.index[s.include])].copy()
    keep["in_win"] = keep.rt.between(RT_MIN, RT_MAX)
    cols = ["Subject", "TaskBlock", "load", "Penalty", "pos", "acc", "rt", "prev_acc", "in_win"]
    keep[cols].to_csv(os.path.join(OUT, "trials.csv"), index=False)
    print(f"trials kept: {len(keep)}  (in RT window: {keep.in_win.mean():.3f})")


if __name__ == "__main__":
    main()
