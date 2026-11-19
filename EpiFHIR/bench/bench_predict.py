"""Prediction on synthetic data: a negative control and a signal-injection
positive control.

Unit: each seizure-related emergency encounter (index). Outcome: another
seizure-related emergency encounter within 365 days. Index encounters whose
follow-up ends (death or end of data) before 365 days without an event are
excluded. In Synthea's epilepsy module the chance of a further seizure visit
does not depend on any patient attribute, so a correct pipeline should give
AUROC close to 0.5 (negative control). We then inject a known effect into a
copy of the outcome and check that the models recover it (positive control).
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SYN = "data/synthea/output/csv"
OUT = "results"
SEIZ, EPI = "128613002", "84757009"
ASM = {"308971": "carbamazepine", "197591": "diazepam", "204892": "clonazepam"}
rng = np.random.default_rng(2026)

rd = lambda n: pd.read_csv(f"{SYN}/{n}.csv", dtype=str)  # noqa: E731
p, e, c, m = rd("patients"), rd("encounters"), rd("conditions"), rd("medications")
for df, cols in ((e, ["START"]), (c, ["START"]), (m, ["START", "STOP"]), (p, ["BIRTHDATE", "DEATHDATE"])):
    for col in cols:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
end_of_data = e.START.max()
s = e[e.REASONCODE == SEIZ].sort_values(["PATIENT", "START"])
pinfo = p.set_index("Id")
rows = []
for pid, grp in s.groupby("PATIENT"):
    times = list(grp.START)
    death = pinfo.at[pid, "DEATHDATE"]
    stop = min(end_of_data, death) if pd.notna(death) else end_of_data
    pc = c[c.PATIENT == pid]
    pm = m[m.PATIENT == pid]
    for k, t in enumerate(times):
        nxt = times[k + 1] if k + 1 < len(times) else None
        event = nxt is not None and (nxt - t).days <= 365
        if not event and (stop - t).days < 365:
            continue  # censored before the outcome window closes
        prior_conds = pc[pc.START < t]
        active_asm = pm[(pm.CODE.isin(ASM)) & (pm.START < t) & (pm.STOP.isna() | (pm.STOP > t))]
        rows.append(dict(
            patient=pid, y=int(event),
            age=(t - pinfo.at[pid, "BIRTHDATE"]).days / 365.25,
            male=int(pinfo.at[pid, "GENDER"] == "M"),
            income=float(pinfo.at[pid, "INCOME"]),
            hispanic=int(pinfo.at[pid, "ETHNICITY"] == "hispanic"),
            white=int(pinfo.at[pid, "RACE"] == "white"),
            prior_seizure_visits=k,
            days_since_last=(t - times[k - 1]).days if k else 9999,
            epilepsy_dx=int((prior_conds.CODE == EPI).any()),
            n_prior_conditions=int(prior_conds.CODE.nunique()),
            on_asm=int(len(active_asm) > 0),
            **{f"asm_{v}": int((active_asm.CODE == code).any()) for code, v in ASM.items()}))
D = pd.DataFrame(rows)
X = D.drop(columns=["patient", "y"]).astype(float)
X["days_since_last"] = np.log1p(X["days_since_last"])
groups = D.patient.values


def models():
    return {"Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                     colsample_bytree=0.8, eval_metric="logloss", n_jobs=2, random_state=0)}


def oof(model, X, y, seed=0):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pr = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        mdl = model()
        mdl.fit(X.iloc[tr], y[tr])
        pr[te] = mdl.predict_proba(X.iloc[te])[:, 1]
    return pr


def cluster_boot_ci(y, pr, B=1000):
    pats = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in pats}
    vals = []
    for _ in range(B):
        pick = np.concatenate([idx[g] for g in rng.choice(pats, len(pats))])
        if len(np.unique(y[pick])) < 2:
            continue
        vals.append(roc_auc_score(y[pick], pr[pick]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluate(y, label, n_perm=50):
    res = {}
    for name in models():
        mk = lambda name=name: models()[name]  # noqa: E731
        pr = oof(mk, X, y)
        auc = roc_auc_score(y, pr)
        npm = n_perm if name.startswith('Logistic') else max(1, n_perm // 5)
        perm = [roc_auc_score(yp, oof(mk, X, yp, seed=i + 1)) for i, yp in
                enumerate(rng.permuted(np.tile(y, (npm, 1)), axis=1))] if n_perm else []
        res[name] = dict(auroc=float(auc), ci=cluster_boot_ci(y, pr),
                         brier=float(np.mean((pr - y) ** 2)), base_rate_brier=float(y.mean() * (1 - y.mean())),
                         perm_mean=float(np.mean(perm)) if perm else None,
                         perm_p=float((1 + sum(a >= auc for a in perm)) / (1 + len(perm))) if perm else None)
        np.save(f"{OUT}/oof_{label}_{name.split()[0].lower()}.npy", np.vstack([y, pr]))
        print(label, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()})
    return res


y = D.y.values
neg = evaluate(y, "negative")

# positive control: inject a known effect of prior seizure visits and age
z1 = (X.prior_seizure_visits - X.prior_seizure_visits.mean()) / X.prior_seizure_visits.std()
z2 = (X.age - X.age.mean()) / X.age.std()
lin = np.log(y.mean() / (1 - y.mean())) + 1.2 * z1 + 0.6 * z2
p_true = 1 / (1 + np.exp(-lin))
y_inj = rng.binomial(1, p_true)
oracle = float(roc_auc_score(y_inj, p_true))
pos = evaluate(y_inj, "positive", n_perm=20)

out = dict(index_encounters=int(len(D)), patients=int(D.patient.nunique()), positives=int(y.sum()),
           prevalence=float(y.mean()), features=list(X.columns), negative_control=neg,
           positive_control=dict(effects={"prior_seizure_visits_z": 1.2, "age_z": 0.6},
                                 prevalence=float(y_inj.mean()), oracle_auroc=oracle, models=pos))
json.dump(out, open(f"{OUT}/prediction.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("index_encounters", "patients", "positives", "prevalence")}), "oracle", oracle)
