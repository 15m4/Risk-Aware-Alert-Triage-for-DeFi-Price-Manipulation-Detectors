"""Reproduce original results and compute all new analyses requested by reviewers."""
import numpy as np, pandas as pd, json, sys
from scipy.stats import spearmanr

df = pd.read_csv("incidents.csv")
RNG_SEED = 20261017
B = 10000

# ---------------- policies (return list of row indices in queue order) -------------
def order_fifo(d):
    return list(d.sort_values(["date", "idx"], kind="mergesort").index)

def order_sev(d, loss_col="loss_usd"):
    return list(d.sort_values([loss_col, "idx"], ascending=[False, True], kind="mergesort").index)

def order_vrs(d, loss_col="loss_usd"):
    return list(d.sort_values(["rank", loss_col, "idx"], ascending=[True, False, True], kind="mergesort").index)

def cutoff(losses, tau):
    s = np.sort(np.asarray(losses))[::-1]
    k = int(np.floor(len(s) * tau))
    k = max(k, 1)
    return s[k - 1]

def order_hybrid(d, tau, loss_col="loss_usd", c=None):
    if c is None:
        c = cutoff(d[loss_col], tau)
    ov = d[d[loss_col] >= c]
    rem = d[d[loss_col] < c]
    return order_sev(ov, loss_col) + order_vrs(rem, loss_col)

def order_recw(d, loss_col="loss_usd"):
    # recurrence-weighted loss: loss x (1/RSF rank)  (rank 1 = most recurrent / highest priority)
    s = d[loss_col] / d["rank"]
    return list(d.assign(_s=s).sort_values(["_s", "idx"], ascending=[False, True], kind="mergesort").index)

def order_recw_freq(d, loss_col="loss_usd"):
    # recurrence-weighted loss using in-corpus class frequency as recurrence weight
    f = d.groupby("vclass")["idx"].transform("count")
    s = d[loss_col] * f
    return list(d.assign(_s=s).sort_values(["_s", "idx"], ascending=[False, True], kind="mergesort").index)

def order_classexp(d):
    # class-expected-loss ranking: fitted leave-one-out median loss of the incident's class
    # (uses only class label, i.e., information available before loss is known)
    est = []
    for i, r in d.iterrows():
        others = d[(d["vclass"] == r["vclass"]) & (d.index != i)]["loss_usd"]
        est.append(others.median() if len(others) else d[d.index != i]["loss_usd"].median())
    return list(d.assign(_e=est).sort_values(["_e", "idx"], ascending=[False, True], kind="mergesort").index)

def lwp(d, order):
    pos = {ix: p + 1 for p, ix in enumerate(order)}
    l = d["loss_usd"]
    return float(sum(l[i] * pos[i] for i in d.index) / l.sum())

def topk(d, order, k):
    return float(d.loc[order[:k], "loss_usd"].sum() / d["loss_usd"].sum())

def positions(order):
    return {ix: p + 1 for p, ix in enumerate(order)}

POL = {
    "FIFO": order_fifo,
    "Severity-only": order_sev,
    "VRS-Triage": order_vrs,
    "RecW (loss/rank)": order_recw,
    "RecW (loss x freq)": order_recw_freq,
    "Class-expected loss (LOO)": order_classexp,
    "Hybrid 5%": lambda d: order_hybrid(d, 0.05),
    "Hybrid 10%": lambda d: order_hybrid(d, 0.10),
}

def summarize(d, label):
    out = {"subset": label, "n": len(d)}
    for name, f in POL.items():
        o = f(d)
        out[name] = {"LWP": round(lwp(d, o), 2),
                     "top5": round(100 * topk(d, o, 5), 1),
                     "top10": round(100 * topk(d, o, 10), 1),
                     "top20": round(100 * topk(d, o, 20), 1)}
    rho, _ = spearmanr(d["rank"], d["loss_usd"])
    out["spearman"] = round(float(rho), 3)
    return out

def spearman_stats(d, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    r = d["rank"].values; l = d["loss_usd"].values
    rho = spearmanr(r, l)[0]
    boots = []
    for _ in range(B):
        ix = rng.integers(0, len(d), len(d))
        v = spearmanr(r[ix], l[ix])[0]
        if not np.isnan(v):
            boots.append(v)
    perm = sum(abs(spearmanr(rng.permutation(r), l)[0]) >= abs(rho) for _ in range(B))
    return {"rho": round(float(rho), 3),
            "ci": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            "perm_p": (perm + 1) / (B + 1)}

def paired_bootstrap(d, a, b, seed=RNG_SEED):
    """P(policy a has higher (worse) LWP than policy b), CI on LWP_a - LWP_b."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(B):
        ix = rng.integers(0, len(d), len(d))
        s = d.iloc[ix].reset_index(drop=True)
        s["idx"] = np.arange(len(s))  # unique tie-break ids for duplicates
        diffs.append(lwp(s, POL[a](s)) - lwp(s, POL[b](s)))
    diffs = np.array(diffs)
    return {"P_worse": round(100 * float((diffs > 0).mean()), 1),
            "ci": [round(float(np.percentile(diffs, 2.5)), 2), round(float(np.percentile(diffs, 97.5)), 2)]}

def scl5_stats(d):
    res = {}
    m = d["vclass"] == "SCL5"
    for name in ["Severity-only", "VRS-Triage", "Hybrid 10%", "RecW (loss/rank)", "Class-expected loss (LOO)", "FIFO"]:
        p = positions(POL[name](d))
        ps = [p[i] for i in d[m].index]
        top10 = POL[name](d)[:10]
        res[name] = {"avg": round(float(np.mean(ps)), 1), "worst": int(max(ps)),
                     "classes_top10": sorted(set(d.loc[top10, "vclass"]))}
    return res

if __name__ == "__main__":
    R = {}
    subsets = {"all": df, "H": df[df.confidence == "H"], "H+M": df[df.confidence != "L"]}
    for k, d in subsets.items():
        d = d.copy()
        R[k] = summarize(d, k)
        R[k]["spearman_full"] = spearman_stats(d)
        R[k]["boot_VRS_vs_FIFO"] = paired_bootstrap(d, "VRS-Triage", "FIFO")
        R[k]["boot_VRS_vs_Sev"] = paired_bootstrap(d, "VRS-Triage", "Severity-only")
        R[k]["boot_RecW_vs_VRS"] = paired_bootstrap(d, "RecW (loss/rank)", "VRS-Triage")
        R[k]["boot_H10_vs_FIFO"] = paired_bootstrap(d, "Hybrid 10%", "FIFO")
        R[k]["scl5"] = scl5_stats(d)
        R[k]["class_counts"] = d["vclass"].value_counts().to_dict()
    json.dump(R, open("results_main.json", "w"), indent=1, default=str)
    print(json.dumps(R, indent=1, default=str))
