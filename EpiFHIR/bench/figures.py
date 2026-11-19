"""Draw all manuscript figures from saved result files (no hand-entered numbers)."""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

R = "results"
F = "figures"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
C = {"ours": "#1b6ca8", "fuzzy": "#e08a2c", "tfidf": "#6a9f58", "exact": "#9a9a9a", "sqlite": "#b0413e",
     "pg": "#1b6ca8"}

A = json.load(open(f"{R}/alignment.json"))["benchmarkA"]

# ---- Figure 2: alignment comparison + tau sweep
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
names = ["EpiAlign (full)", "Word TF-IDF retrieval", "Fuzzy matching (rapidfuzz WRatio)", "Exact string match"]
short = ["EpiAlign", "Word TF-IDF", "Fuzzy (WRatio)", "Exact match"]
cols = [C["ours"], C["tfidf"], C["fuzzy"], C["exact"]]
x = np.arange(3)
w = 0.2
for i, (n, s, col) in enumerate(zip(names, short, cols)):
    m = A["systems"][n]
    ax[0].bar(x + (i - 1.5) * w, [m["precision"], m["recall"], 100 * m["f1"]], w, color=col, label=s)
ax[0].set_xticks(x, ["Precision", "Recall", "F1 (x100)"])
ax[0].set_ylabel("%")
ax[0].set_ylim(0, 105)
ax[0].legend(frameon=False, fontsize=7, loc="upper right")
ax[0].set_title("a  Held-out synonym test set (n=%d)" % A["test"], loc="left", fontsize=9)
sw = A["test_sweep"]
t = [r["tau"] for r in sw]
ax[1].plot(t, [r["precision"] for r in sw], "-o", ms=3, color=C["ours"], label="Precision")
ax[1].plot(t, [r["recall"] for r in sw], "-s", ms=3, color=C["fuzzy"], label="Recall")
ax[1].plot(t, [r["coverage"] for r in sw], "--", color="#777", label="Coverage")
ax[1].plot(t, [100 * r["f1"] for r in sw], "-^", ms=3, color=C["tfidf"], label="F1 (x100)")
ax[1].axvline(A["tau_selected"], color="k", lw=0.8, ls=":")
ax[1].text(A["tau_selected"] + 0.01, 2, r"$\tau$ chosen on tuning set", fontsize=7)
ax[1].set_xlabel(r"Similarity threshold $\tau$")
ax[1].set_ylabel("%")
ax[1].set_ylim(0, 105)
ax[1].legend(frameon=False, fontsize=7, loc="lower left")
ax[1].set_title("b  Threshold sweep (test set)", loc="left", fontsize=9)
fig.tight_layout()
fig.savefig(f"{F}/fig_alignment.pdf")

# ---- Figure 3: robustness
rob = A["robustness"]
fig, ax = plt.subplots(figsize=(7.2, 2.5))
perts = ["case_punct", "word_order", "inflection", "typo"]
lab = ["Case and punctuation", "Word order reversed", "Inflection", "Single typo"]
base = {"EpiAlign": A["systems"]["EpiAlign (full)"]["recall"],
        "WordTFIDF": A["systems"]["Word TF-IDF retrieval"]["recall"],
        "Fuzzy": A["systems"]["Fuzzy matching (rapidfuzz WRatio)"]["recall"],
        "Exact": A["systems"]["Exact string match"]["recall"]}
sysk = ["EpiAlign", "WordTFIDF", "Fuzzy", "Exact"]
x = np.arange(len(perts) + 1)
for i, (k, col, s) in enumerate(zip(sysk, cols, short)):
    vals = [base[k]] + [rob[p][k] for p in perts]
    ax.bar(x + (i - 1.5) * w, vals, w, color=col, label=s)
ax.set_xticks(x, ["Unperturbed"] + [l.replace(" and ", " and\n").replace(" reversed", "\nreversed") for l in lab])
ax.set_ylabel("Recall (%)")
ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper right")
ax.set_ylim(0, 65)
fig.tight_layout()
fig.savefig(f"{F}/fig_robustness.pdf")

# ---- Figure 4: storage
S = json.load(open(f"{R}/storage.json"))
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
for kind, col, lab_ in (("sqlite", C["sqlite"], "SQLite"), ("postgresql", C["pg"], "PostgreSQL")):
    rr = [r for r in S if r["backend"] == kind]
    n = [r["patients"] for r in rr]
    ax[0].plot(n, [r["load_rows_per_s"] / 1000 for r in rr], "-o", color=col, label=lab_)
    ax[1].plot(n, [r["cohort_query_ms"] for r in rr], "-o", color=col, label=lab_)
    ax[2].plot(n, [r["concurrent_commits_per_s"] for r in rr], "-o", color=col, label=lab_)
ax[0].set_ylabel("Bulk load (1,000 rows/s)")
ax[1].set_ylabel("Ontology cohort query (ms)")
ax[2].set_ylabel("Concurrent commits/s (4 threads)")
for a_, ttl in zip(ax, "abc"):
    a_.set_xlabel("Patients")
    a_.set_xscale("log")
    a_.minorticks_off()
    a_.set_xticks([2095, 10475, 41900], ["2.1k", "10.5k", "41.9k"])
    a_.set_title(ttl, loc="left", fontsize=9)
ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(f"{F}/fig_storage.pdf")

# ---- Figure 5: negative / positive control ROC (empirical, out-of-fold)
P = json.load(open(f"{R}/prediction.json"))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
for j, (lab_, key) in enumerate((("negative", "negative_control"), ("positive", "positive_control"))):
    for mdl, col in (("logistic", C["ours"]), ("xgboost", C["fuzzy"])):
        y, pr = np.load(f"{R}/oof_{lab_}_{mdl}.npy")
        fpr, tpr, _ = roc_curve(y, pr)
        res = P[key] if key == "negative_control" else P[key]["models"]
        nm = "Logistic regression" if mdl == "logistic" else "XGBoost"
        ax[j].step(fpr, tpr, where="post", color=col, lw=1.2, label=f"{nm} (AUROC {res[nm]['auroc']:.3f})")
    ax[j].plot([0, 1], [0, 1], ":", color="#777", lw=0.8)
    ax[j].set_xlabel("1 - specificity")
    ax[j].set_ylabel("Sensitivity")
    ax[j].legend(frameon=False, fontsize=7, loc="lower right")
ax[0].set_title("a  Negative control", loc="left", fontsize=9)
ax[1].set_title("b  Positive control (oracle AUROC %.3f)" % P["positive_control"]["oracle_auroc"], loc="left", fontsize=9)
fig.tight_layout()
fig.savefig(f"{F}/fig_prediction.pdf")
print("figures written")
