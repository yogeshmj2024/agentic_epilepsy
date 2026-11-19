"""Benchmarks for ontology alignment (all inputs public, fully reproducible).

A  Held-out synonym task: for each EPIO class with >= 1 exact synonym, one synonym
   is removed from the index and used as the query; the gold answer is its class.
   Queries are split 50/50 into a tuning set (threshold selection) and a test set.
A' Robustness: the test queries are perturbed (typo, word order, inflection,
   case/punctuation) and re-aligned.
B  ICD-10-CM G40/G41 code titles aligned to EPIO; gold = EPIO classes that
   cross-reference the code or its nearest ancestor code.
"""
import json
import random
import sys
import time
from math import sqrt

import numpy as np
import simple_icd_10_cm as icd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, ".")
from epifhir.align import Aligner, normalise, surface_forms  # noqa: E402
from epifhir.ontology import load_ontology, xref_index  # noqa: E402

OWL = "data/EpilepsyOntology/release/EPIO_merged.owl"
OUT = "results"
rng = random.Random(7)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def metrics(preds, golds):
    n = len(golds)
    made = [(p, g) for p, g in zip(preds, golds) if p is not None]
    corr = sum(1 for p, g in made if p in g)
    prec = corr / len(made) if made else 0.0
    rec = corr / n
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n=n, predicted=len(made), correct=corr, coverage=100 * len(made) / n,
                precision=100 * prec, recall=100 * rec, f1=f1,
                precision_ci=wilson(corr, len(made)), recall_ci=wilson(corr, n))


# ---------------------------------------------------------------- baselines
class ExactBaseline:
    def __init__(self, forms):
        self.d = {}
        for t, iri in forms:
            self.d.setdefault(t.strip().lower(), set()).add(iri)

    def run(self, q, thr=None):
        h = self.d.get(q.strip().lower())
        return next(iter(h)) if h and len(h) == 1 else None


class FuzzyBaseline:
    """rapidfuzz WRatio nearest neighbour over all surface forms."""

    def __init__(self, forms):
        self.texts = [t.lower() for t, _ in forms]
        self.iris = [i for _, i in forms]

    def scores(self, q):
        r = process.extractOne(q.lower(), self.texts, scorer=fuzz.WRatio)
        return self.iris[r[2]], r[1] / 100.0

    def run(self, q, thr):
        iri, s = self.scores(q)
        return iri if s >= thr else None


class WordTfidfBaseline:
    def __init__(self, forms):
        self.iris = [i for _, i in forms]
        self.v = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        self.M = self.v.fit_transform([t.lower() for t, _ in forms])

    def scores(self, q):
        s = (self.M @ self.v.transform([q.lower()]).T).toarray().ravel()
        b = int(np.argmax(s))
        return self.iris[b], float(s[b])

    def run(self, q, thr):
        iri, s = self.scores(q)
        return iri if s >= thr else None


# ---------------------------------------------------------------- perturbations
def typo(q):
    ws = q.split()
    idx = max(range(len(ws)), key=lambda i: len(ws[i]))
    w = ws[idx]
    if len(w) < 5:
        return q
    j = rng.randrange(1, len(w) - 2)
    ws[idx] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(ws)


def reorder(q):
    ws = q.split()
    return " ".join(reversed(ws)) if len(ws) > 1 else q


def inflect(q):
    ws = q.split()
    w = ws[-1]
    ws[-1] = w[:-1] + "ies" if w.endswith("y") else (w if w.endswith("s") else w + "s")
    return " ".join(ws)


def casepunct(q):
    return q.upper().replace(" ", "-", 1) + "."


PERT = {"typo": typo, "word_order": reorder, "inflection": inflect, "case_punct": casepunct}


def main():
    t0 = time.time()
    classes = load_ontology(OWL)
    n_epso = sum(c.from_epso for c in classes.values())
    n_syn = sum(len(c.exact) for c in classes.values())

    # ---------------- Benchmark A construction
    queries, drop = [], {}
    for c in sorted(classes.values(), key=lambda c: c.iri):
        if c.exact:
            s = rng.choice(c.exact)
            queries.append((s, c.iri))
            drop.setdefault(c.iri, set()).add(s)
    forms = surface_forms(classes, drop=drop)
    norm_owner = {}
    for t, iri in forms:
        norm_owner.setdefault(normalise(t), set()).add(iri)
    kept, ambiguous = [], 0
    for q, iri in queries:
        owners = norm_owner.get(normalise(q), set())
        if owners and iri not in owners:
            ambiguous += 1  # identical to another class's form: gold undefined
            continue
        kept.append((q, iri, bool(owners)))  # trivial if normalises to own form
    rng.shuffle(kept)
    half = len(kept) // 2
    tune, test = kept[:half], kept[half:]

    # ---------------- systems
    def ours(**kw):
        return Aligner(forms, **kw)

    full = ours(tau=0.0)
    tune_scores = [full.align(q) for q, _, _ in tune]
    taus = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]
    sweep = []
    for t in taus:
        preds = [p if s >= t else None for p, s, _ in tune_scores]
        m = metrics(preds, [{g} for _, g, _ in tune])
        sweep.append(dict(tau=t, **{k: m[k] for k in ("coverage", "precision", "recall", "f1")}))
    tau_star = max(sweep, key=lambda r: r["f1"])["tau"]

    def eval_aligner(al, qs, tau):
        t1 = time.perf_counter()
        res = [al.align(q) for q, _, _ in qs]
        ms = 1000 * (time.perf_counter() - t1) / len(qs)
        preds = [p if (p is not None and s >= tau) else None for p, s, _ in res]
        return metrics(preds, [{g} for _, g, _ in qs]), ms, res

    systems = {}
    m, ms, res_full = eval_aligner(full, test, tau_star)
    systems["EpiAlign (full)"] = dict(m, ms_per_query=ms)

    def tune_thr(base, grid):
        best = None
        sc = [base.scores(q) for q, _, _ in tune]
        for t in grid:
            preds = [i if s >= t else None for i, s in sc]
            f = metrics(preds, [{g} for _, g, _ in tune])["f1"]
            if best is None or f > best[1]:
                best = (t, f)
        return best[0]

    ex = ExactBaseline(forms)
    t1 = time.perf_counter()
    preds = [ex.run(q) for q, _, _ in test]
    systems["Exact string match"] = dict(metrics(preds, [{g} for _, g, _ in test]),
                                         ms_per_query=1000 * (time.perf_counter() - t1) / len(test))
    for name, B, grid in [("Fuzzy matching (rapidfuzz WRatio)", FuzzyBaseline, np.arange(0.5, 1.0, 0.025)),
                          ("Word TF-IDF retrieval", WordTfidfBaseline, np.arange(0.2, 0.95, 0.025))]:
        b = B(forms)
        thr = tune_thr(b, grid)
        t1 = time.perf_counter()
        preds = [b.run(q, thr) for q, _, _ in test]
        systems[name] = dict(metrics(preds, [{g} for _, g, _ in test]), threshold=float(thr),
                             ms_per_query=1000 * (time.perf_counter() - t1) / len(test))

    # ---------------- ablations (same tau)
    abl = {}
    for name, kw in [("without lemmatisation", dict(lemmatise=False)),
                     ("without trie (full scan)", dict(use_trie=False)),
                     ("without Stage 3 (token Jaccard)", dict(use_stage3=False))]:
        al = ours(tau=0.0, **kw)
        m, ms, _ = eval_aligner(al, test, tau_star)
        abl[name] = dict(m, ms_per_query=ms)
    rel_forms = forms + [(r, c.iri) for c in classes.values() for r in c.related]
    al = Aligner(rel_forms, tau=0.0)
    m, ms, _ = eval_aligner(al, test, tau_star)
    abl["with related synonyms indexed"] = dict(m, ms_per_query=ms)

    # non-trivial subset
    nt = [q for q in test if not q[2]]
    m_nt, _, _ = eval_aligner(full, nt, tau_star)
    routes = {}
    for _, _, r in res_full:
        routes[r] = routes.get(r, 0) + 1

    # full test-set tau sweep (for figure)
    test_sweep = []
    for t in taus:
        preds = [p if s >= t else None for p, s, _ in res_full]
        mm = metrics(preds, [{g} for _, g, _ in test])
        test_sweep.append(dict(tau=t, **{k: mm[k] for k in ("coverage", "precision", "recall", "f1")}))

    # ---------------- robustness
    robust = {}
    for pname, fn in PERT.items():
        pq = [(fn(q), g, False) for q, g, _ in test]
        row = {}
        for sname, runner in [("EpiAlign", lambda q: (lambda r: r[0] if r[1] >= tau_star else None)(full.align(q))),
                              ("Exact", ex.run),
                              ("Fuzzy", lambda q, b=FuzzyBaseline(forms), t=systems["Fuzzy matching (rapidfuzz WRatio)"]["threshold"]: b.run(q, t)),
                              ("WordTFIDF", lambda q, b=WordTfidfBaseline(forms), t=systems["Word TF-IDF retrieval"]["threshold"]: b.run(q, t))]:
            mm = metrics([runner(q) for q, _, _ in pq], [{g} for _, g, _ in pq])
            row[sname] = mm["recall"]
        robust[pname] = row

    # ---------------- Benchmark B: ICD-10-CM titles (hierarchy-consistency check)
    # EPIO ICD-10-CM xrefs mean "classified under", not equivalence, so we do not
    # treat them as gold labels. A prediction is *consistent* if the predicted
    # class cross-references the code itself or one of its ancestor codes;
    # *unverifiable* if the predicted class has no ICD-10-CM xref.
    codes = [c for c in icd.get_all_codes(False) if c.startswith(("G40", "G41"))]
    forms_all = surface_forms(classes)
    fullB = Aligner(forms_all, tau=0.0)
    exB, fzB, tfB = ExactBaseline(forms_all), FuzzyBaseline(forms_all), WordTfidfBaseline(forms_all)
    def ancestors(code):
        return {code[:k] for k in range(3, len(code) + 1)}
    def judge(pred, code):
        if pred is None:
            return "abstain"
        x = classes[pred].xrefs.get("ICD10CM")
        if not x:
            return "unverifiable"
        return "consistent" if x & ancestors(code) else "inconsistent"
    runners = {"EpiAlign (full)": lambda d: (lambda r: r[0] if r[1] >= tau_star else None)(fullB.align(d)),
               "Exact string match": exB.run,
               "Fuzzy matching (rapidfuzz WRatio)": lambda d: fzB.run(d, systems["Fuzzy matching (rapidfuzz WRatio)"]["threshold"]),
               "Word TF-IDF retrieval": lambda d: tfB.run(d, systems["Word TF-IDF retrieval"]["threshold"])}
    sysB, examplesB = {}, []
    for name, fn in runners.items():
        cnt = {"consistent": 0, "inconsistent": 0, "unverifiable": 0, "abstain": 0}
        for code in codes:
            pr = fn(icd.get_description(code))
            j = judge(pr, code)
            cnt[j] += 1
            if name == "EpiAlign (full)":
                examplesB.append(dict(code=code, title=icd.get_description(code),
                                      predicted=classes[pr].label if pr else None, judgement=j))
        verif = cnt["consistent"] + cnt["inconsistent"]
        sysB[name] = dict(cnt, aligned=len(codes) - cnt["abstain"],
                          consistency_of_verifiable=(100 * cnt["consistent"] / verif) if verif else None,
                          consistency_ci=wilson(cnt["consistent"], verif))
    Bg = codes
    out = dict(
        ontology=dict(classes=len(classes), exact_synonyms=n_syn, from_epso=n_epso,
                      surface_forms_indexed=len(forms),
                      xref_counts={s: sum(1 for c in classes.values() if s in c.xrefs)
                                   for s in ("SNOMED", "ICD10CM", "ICD9CM", "UMLS", "MESH")}),
        benchmarkA=dict(queries_total=len(queries), excluded_ambiguous=ambiguous,
                        tune=len(tune), test=len(test), test_nontrivial=len(nt),
                        tau_selected=tau_star, tune_sweep=sweep, test_sweep=test_sweep,
                        systems=systems, ablations=abl, nontrivial=m_nt, routes=routes,
                        robustness=robust),
        benchmarkB=dict(codes_total=len(codes), systems=sysB,
                        examples=examplesB),
        runtime_s=time.time() - t0)
    json.dump(out, open(f"{OUT}/alignment.json", "w"), indent=1, default=list)
    print(json.dumps({k: out[k] for k in ("ontology",)}, indent=1, default=list))
    print("tau*", tau_star, "test", len(test), "nontrivial", len(nt), "amb", ambiguous)
    for k, v in systems.items():
        print(f"{k:38s} cov {v['coverage']:.1f} P {v['precision']:.1f} R {v['recall']:.1f} F1 {v['f1']:.3f} {v['ms_per_query']:.2f}ms")
    for k, v in abl.items():
        print(f"  {k:36s} P {v['precision']:.1f} R {v['recall']:.1f} F1 {v['f1']:.3f} {v['ms_per_query']:.2f}ms")
    print("nontrivial", {k: round(m_nt[k], 2) for k in ('precision', 'recall', 'f1')}, routes)
    print("robust", json.dumps(robust))
    print("B", len(codes), json.dumps(sysB))


if __name__ == "__main__":
    main()
