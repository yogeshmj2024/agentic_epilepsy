"""Write LaTeX table bodies directly from result files."""
import json, sqlite3
R = "results"; T = "paper/tables"
A = json.load(open(f"{R}/alignment.json")); a = A["benchmarkA"]
F = json.load(open(f"{R}/fhir.json")); S = json.load(open(f"{R}/storage.json")); P = json.load(open(f"{R}/prediction.json"))
ci = lambda c: f"({c[0]:.1f}--{c[1]:.1f})"
rows = []
for name, lab in [("EpiAlign (full)", "\\textbf{EpiAlign (this work)}"), ("Word TF-IDF retrieval", "Word TF-IDF retrieval"),
                  ("Fuzzy matching (rapidfuzz WRatio)", "Fuzzy matching (WRatio)"), ("Exact string match", "Exact string match")]:
    m = a["systems"][name]
    thr = f"{m['threshold']:.2f}" if "threshold" in m else (f"{a['tau_selected']:.2f}" if name.startswith("EpiAlign") else "--")
    rows.append(f"{lab} & {thr} & {m['coverage']:.1f} & {m['precision']:.1f} {ci(m['precision_ci'])} & {m['recall']:.1f} {ci(m['recall_ci'])} & {m['f1']:.3f} & {m['ms_per_query']:.2f} \\\\")
open(f"{T}/align_main.tex", "w").write("\n".join(rows))
full = a["systems"]["EpiAlign (full)"]
rows = [f"Full pipeline & {full['coverage']:.1f} & {full['precision']:.1f} & {full['recall']:.1f} & {full['f1']:.3f} & {full['ms_per_query']:.2f} \\\\ \\midrule"]
for k, lab in [("without lemmatisation", "Without lemmatisation"), ("without trie (full scan)", "Without trie (scan all forms)"),
               ("without Stage 3 (token Jaccard)", "Without Stage 3 (token Jaccard)"), ("with related synonyms indexed", "Adding related synonyms to index")]:
    m = a["ablations"][k]
    rows.append(f"{lab} & {m['coverage']:.1f} & {m['precision']:.1f} & {m['recall']:.1f} & {m['f1']:.3f} & {m['ms_per_query']:.2f} \\\\")
open(f"{T}/align_ablation.tex", "w").write("\n".join(rows))
rows = []
for name, lab in [("EpiAlign (full)", "EpiAlign"), ("Word TF-IDF retrieval", "Word TF-IDF"), ("Fuzzy matching (rapidfuzz WRatio)", "Fuzzy (WRatio)"), ("Exact string match", "Exact match")]:
    m = A["benchmarkB"]["systems"][name]
    rows.append(f"{lab} & {m['aligned']} & {m['abstain']} & {m['consistent']} & {m['inconsistent']} & {m['unverifiable']} \\\\")
open(f"{T}/icd.tex", "w").write("\n".join(rows))
rc = F["resources"]
rows = [f"{k} & {v:,} \\\\" for k, v in rc.items()] + [f"\\midrule Total & {sum(rc.values()):,} \\\\"]
open(f"{T}/resources.tex", "w").write("\n".join(rows))
c = F["concordance"]
rows = [f"{k} & {v['ours']:,} & {v['native']:,} & {v['both']:,} & {v['jaccard']:.3f} & {v['patients_identical']:,}/{v['patients_compared']:,} \\\\" for k, v in c.items()]
open(f"{T}/concordance.tex", "w").write("\n".join(rows))
sanity = [l.split("\t") for l in open(f"{R}/sanity_validation.tsv")]
lab = {"00_unmodified": "None (exported bundle)", "01_missing_required_status": "Required element removed (MedicationRequest.status)",
       "02_invalid_date": "Impossible date (2021-13-45)", "03_unknown_element": "Undefined element added",
       "04_wrong_datatype": "Wrong datatype (birthDate as number)", "05_invalid_status_code": "Status value outside required value set"}
rows = []
for f, e, w, ms, *_ in sanity:
    k = f.split("/")[-1][:-5]
    rows.append(f"{lab[k]} & {e} & {'yes' if int(e) > 0 else ('--' if k.startswith('00') else 'no')} \\\\")
open(f"{T}/sanity.tex", "w").write("\n".join(rows))
rows = []
for r in S:
    rows.append(f"{'SQLite' if r['backend']=='sqlite' else 'PostgreSQL'} & {r['patients']:,} & {r['rows']:,} & {r['load_rows_per_s']/1000:.1f} & {r['cohort_query_ms']:.1f} & {r['record_ms_p50']:.2f} / {r['record_ms_p95']:.2f} & {r['concurrent_commits_per_s']:,.0f} ({r['concurrent_errors']}) \\\\")
open(f"{T}/storage.tex", "w").write("\n".join(rows))
rows = []
for ctl, key in (("Negative", "negative_control"), ("Positive", "positive_control")):
    d = P[key] if ctl == "Negative" else P[key]["models"]
    for m in ("Logistic regression", "XGBoost"):
        x = d[m]
        rows.append(f"{ctl} & {m} & {x['auroc']:.3f} {tuple(round(v,3) for v in x['ci'])} & {x['brier']:.3f} & {x['base_rate_brier']:.3f} & {x['perm_mean']:.3f} \\\\".replace("(0.", "(0.").replace(", ", "--"))
open(f"{T}/prediction.tex", "w").write("\n".join(rows))
print("sqlite", sqlite3.sqlite_version)
