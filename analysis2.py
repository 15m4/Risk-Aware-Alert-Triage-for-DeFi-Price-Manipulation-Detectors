import numpy as np, pandas as pd, json
from analysis import *

out = {}
d = df.copy()
print("total loss", d.loss_usd.sum(), "VRS top20 exact", topk(d, order_vrs(d), 20))

# (a) tau sweep with named-incident positions and SCL5 wait
names = ["CreamFinance", "BonqDAO", "PancakeBunny", "Harvest"]
nm = {n: d.index[d.name == n][0] for n in names}
rows = []
def rowfor(label, o, c=None):
    p = positions(o)
    s5 = [p[i] for i in d[d.vclass == "SCL5"].index]
    r = {"policy": label, "cutoff": c, "LWP": round(lwp(d, o), 2),
         "top5": round(100*topk(d,o,5),1), "top10": round(100*topk(d,o,10),1), "top20": round(100*topk(d,o,20),1),
         "SCL5avg": round(np.mean(s5), 1), "SCL5worst": max(s5)}
    r.update({n: p[nm[n]] for n in names})
    return r
rows.append(rowfor("FIFO", order_fifo(d)))
rows.append(rowfor("Severity", order_sev(d)))
rows.append(rowfor("VRS", order_vrs(d)))
rows.append(rowfor("RecW", order_recw(d)))
for t in [0.05, 0.10, 0.15, 0.20, 0.30]:
    rows.append(rowfor(f"Hybrid {int(t*100)}%", order_hybrid(d, t), cutoff(d.loss_usd, t)))
out["sweep"] = rows
print(pd.DataFrame(rows).to_string())

# hybrid sweep on H subset
dh = df[df.confidence == "H"].copy()
hs = []
for t in [0.05, 0.10, 0.15, 0.20, 0.30]:
    o = order_hybrid(dh, t); p = positions(o)
    s5 = [p[i] for i in dh[dh.vclass == "SCL5"].index]
    hs.append({"tau": t, "cut": cutoff(dh.loss_usd, t), "LWP": round(lwp(dh, o), 2), "SCL5avg": round(np.mean(s5),1)})
out["sweep_H"] = hs
print(pd.DataFrame(hs).to_string())

# (b) noisy loss at alert time
rng = np.random.default_rng(7)
noise = {}
for sigma in [0.5, 1.0, 2.0]:
    res = {"Severity": [], "Hybrid 10%": [], "Hybrid 5%": [], "RecW": []}
    for _ in range(2000):
        dn = d.copy()
        dn["est"] = d.loss_usd * np.exp(rng.normal(0, sigma, len(d)))
        res["Severity"].append(lwp(d, order_sev(dn, "est")))
        res["RecW"].append(lwp(d, order_recw(dn, "est")))
        res["Hybrid 10%"].append(lwp(d, order_hybrid(dn, 0.10, "est")))
        res["Hybrid 5%"].append(lwp(d, order_hybrid(dn, 0.05, "est")))
    noise[sigma] = {k: {"mean": round(float(np.mean(v)), 2),
                        "p95": round(float(np.percentile(v, 95)), 2),
                        "P_beats_FIFO": round(100*float(np.mean(np.array(v) < 21.41)), 1),
                        "P_beats_VRS": round(100*float(np.mean(np.array(v) < 28.34)), 1)} for k, v in res.items()}
out["noise"] = noise
print(json.dumps(noise, indent=1))

# (c) temporal hold-out: dollar cutoff from 2020-2021 incidents, applied to 2022-2023
tr = d[d.date < "2022-01-01"]; te = d[d.date >= "2022-01-01"].copy()
temp = {"n_train": len(tr), "n_test": len(te),
        "test_FIFO": round(lwp(te, order_fifo(te)), 2), "test_Sev": round(lwp(te, order_sev(te)), 2),
        "test_VRS": round(lwp(te, order_vrs(te)), 2), "rows": []}
for t in [0.10, 0.20, 0.30]:
    c_tr = cutoff(tr.loss_usd, t)
    o = order_hybrid(te, t, c=c_tr)
    o2 = order_hybrid(te, t)
    temp["rows"].append({"tau": t, "c_train": c_tr, "LWP_test_traincut": round(lwp(te, o), 2),
                         "c_test": cutoff(te.loss_usd, t), "LWP_test_oraclecut": round(lwp(te, o2), 2),
                         "n_override_test": int((te.loss_usd >= c_tr).sum())})
out["temporal"] = temp
print(json.dumps(temp, indent=1, default=float))
json.dump(out, open("results_extra.json", "w"), indent=1, default=float)
