"""Cohort description, FHIR R4 export, HAPI validation and cross-implementation
concordance against Synthea's own FHIR exporter."""
import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from epifhir.align import Aligner, surface_forms  # noqa: E402
from epifhir.fhir_export import Exporter, write_json  # noqa: E402
from epifhir.ontology import load_ontology, xref_index  # noqa: E402

SYN = "data/synthea/output"
OUT = "results"
BUNDLES = "results/bundles"
os.makedirs(BUNDLES, exist_ok=True)
SEIZ, EPI, HIST, EEG = "128613002", "84757009", "1290882004", "54550000"
ASM = {"308971": "carbamazepine", "197591": "diazepam", "204892": "clonazepam"}

rd = lambda n: pd.read_csv(f"{SYN}/csv/{n}.csv", dtype=str)  # noqa: E731
pat, enc, cond, proc, med = rd("patients"), rd("encounters"), rd("conditions"), rd("procedures"), rd("medications")
for df in (enc, cond, proc, med):
    df.columns = [c.upper() for c in df.columns]
pat.columns = [c.upper() for c in pat.columns]
enc = enc.rename(columns={"ID": "Id"})

# ---------------------------------------------------------------- cohort
cohort = set(cond.loc[cond.CODE == SEIZ, "PATIENT"])
pat = pat[pat.ID.isin(cohort)].copy()
first_seiz = cond[cond.CODE == SEIZ].groupby("PATIENT").START.min()
pat["age_onset"] = (pd.to_datetime(first_seiz.reindex(pat.ID).values) - pd.to_datetime(pat.BIRTHDATE.values)).days / 365.25
epi_pts = set(cond.loc[cond.CODE == EPI, "PATIENT"])
seiz_enc = enc[(enc.PATIENT.isin(cohort)) & (enc.REASONCODE == SEIZ)]
eeg = proc[(proc.CODE == EEG) & proc.PATIENT.isin(cohort)]
asm = med[med.CODE.isin(ASM) & med.PATIENT.isin(cohort)]
desc = dict(
    patients=len(pat), male=int((pat.GENDER == "M").sum()), female=int((pat.GENDER == "F").sum()),
    age_onset_median=float(np.median(pat.age_onset)), age_onset_iqr=[float(np.percentile(pat.age_onset, 25)), float(np.percentile(pat.age_onset, 75))],
    with_epilepsy_dx=len(epi_pts & cohort), deceased=int(pat.DEATHDATE.notna().sum()),
    race=pat.RACE.value_counts().to_dict(), ethnicity=pat.ETHNICITY.value_counts().to_dict(),
    seizure_encounters=len(seiz_enc), seizure_enc_per_patient_median=float(seiz_enc.groupby("PATIENT").size().reindex(list(cohort), fill_value=0).median()),
    seizure_enc_per_patient_max=int(seiz_enc.groupby("PATIENT").size().max()),
    eeg_procedures=len(eeg), asm_orders=len(asm), asm_by_drug={ASM[k]: int(v) for k, v in asm.CODE.value_counts().items()},
    patients_with_asm=asm.PATIENT.nunique(),
    total_conditions=int(cond.PATIENT.isin(cohort).sum()), total_encounters=int(enc.PATIENT.isin(cohort).sum()),
    total_procedures=int(proc.PATIENT.isin(cohort).sum()), total_medications=int(med.PATIENT.isin(cohort).sum()),
    distinct_condition_codes=int(cond[cond.PATIENT.isin(cohort)].CODE.nunique()))
json.dump(desc, open(f"{OUT}/cohort.json", "w"), indent=1)
print(json.dumps(desc, indent=1)[:1500])

# ---------------------------------------------------------------- export
classes = load_ontology("data/EpilepsyOntology/release/EPIO_merged.owl")
tau = json.load(open(f"{OUT}/alignment.json"))["benchmarkA"]["tau_selected"]
al = Aligner(surface_forms(classes), tau=tau)
ex = Exporter(classes, xref_index(classes, "SNOMED"), al, tau, coded_text_policy="auto")  # behaviour evaluated in the paper
g = {k: df[df.PATIENT.isin(cohort)].groupby("PATIENT") for k, df in
     (("enc", enc), ("cond", cond), ("proc", proc), ("med", med))}
empty = {k: df.iloc[0:0] for k, df in (("enc", enc), ("cond", cond), ("proc", proc), ("med", med))}
get = lambda k, pid: g[k].get_group(pid) if pid in g[k].groups else empty[k]  # noqa: E731
times, rcounts = [], Counter()
ont_by_code = {}
for p in pat.itertuples(index=False):
    t0 = time.perf_counter()
    b = ex.bundle(p.ID, p, get("enc", p.ID), get("cond", p.ID), get("proc", p.ID), get("med", p.ID))
    times.append(time.perf_counter() - t0)
    for e in b["entry"]:
        r = e["resource"]
        rcounts[r["resourceType"]] += 1
        if r["resourceType"] == "Condition":
            cs = r["code"]["coding"]
            ont_by_code.setdefault((cs[0]["code"], cs[0]["display"]), cs[1]["display"] if len(cs) > 1 else None)
    write_json(b, f"{BUNDLES}/{p.ID}.json")
print("export", len(times), "bundles", dict(rcounts), ex.stats)

# ---------------------------------------------------------------- HAPI validation (ours + native)
native_files = {}
for f in glob.glob(f"{SYN}/fhir/*.json"):
    pid = os.path.basename(f).rsplit("_", 1)[-1][:-5]
    native_files[pid] = f
ours_files = sorted(glob.glob(f"{BUNDLES}/*.json"))
import random
nat_all = [native_files[p] for p in sorted(cohort) if p in native_files]
nat_list = sorted(random.Random(5).sample(nat_all, 100))


def hapi(files, tag):
    lst = f"/tmp/{tag}.txt"
    res = []
    for i in range(0, len(files), 200):
        chunk = files[i:i + 200]
        out = subprocess.run(["java", "-Xmx3g", "-cp", "tools/synthea.jar:validator", "Validate", *chunk],
                             capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            f, e, w, ms, msg = (line.split("\t") + [""])[:5]
            res.append(dict(file=f, errors=int(e), warnings=int(w), ms=int(ms), msg=msg))
    json.dump(res, open(f"{OUT}/hapi_{tag}.json", "w"))
    return res


t0 = time.time()
v_ours = json.load(open(f"{OUT}/hapi_ours.json")) if os.path.exists(f"{OUT}/hapi_ours.json") else hapi(ours_files, "ours")
v_nat = hapi(nat_list, "native")
print("hapi done", time.time() - t0)


def summarise(v):
    prof = sum(m["msg"].count("has not been checked because it is unknown") for m in v)
    return dict(bundles=len(v), bundles_zero_errors=sum(1 for m in v if m["errors"] == 0),
                total_errors=sum(m["errors"] for m in v), total_warnings=sum(m["warnings"] for m in v),
                median_ms=float(np.median([m["ms"] for m in v])), sample_msgs=[m["msg"][:300] for m in v if m["errors"]][:5],
                unknown_profile_msgs_in_first5=prof)


# ---------------------------------------------------------------- concordance with native Synthea FHIR
def keyset(bundle, rtype):
    s = set()
    for e in bundle["entry"]:
        r = e["resource"]
        if r["resourceType"] != rtype:
            continue
        if rtype == "Condition":
            s.add((r["code"]["coding"][0]["code"], (r.get("onsetDateTime") or "")[:10]))
        elif rtype == "MedicationRequest":
            cc = r.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("code")
            s.add((cc, (r.get("authoredOn") or "")[:10]))
        elif rtype == "Procedure":
            per = r.get("performedPeriod", {}).get("start") or r.get("performedDateTime") or ""
            s.add((r["code"]["coding"][0]["code"], per[:10]))
    return s


conc = {t: dict(ours=0, native=0, both=0, patients_identical=0) for t in ("Condition", "MedicationRequest", "Procedure")}
npairs = 0
for pid in sorted(cohort):
    if pid not in native_files:
        continue
    npairs += 1
    o = json.load(open(f"{BUNDLES}/{pid}.json"))
    n = json.load(open(native_files[pid]))
    for t in conc:
        a, b = keyset(o, t), keyset(n, t)
        conc[t]["ours"] += len(a)
        conc[t]["native"] += len(b)
        conc[t]["both"] += len(a & b)
        conc[t]["patients_identical"] += int(a == b)
for t, c in conc.items():
    c["jaccard"] = c["both"] / (c["ours"] + c["native"] - c["both"]) if c["ours"] + c["native"] else 1.0
    c["patients_compared"] = npairs

sizes = [os.path.getsize(f) for f in ours_files]
out = dict(cohort=desc, resources=dict(rcounts), ontology_coding=ex.stats,
           ontology_by_condition_code={f"{k[0]}|{k[1]}": v for k, v in ont_by_code.items()},
           export_ms_per_patient=dict(median=1000 * float(np.median(times)), p95=1000 * float(np.percentile(times, 95))),
           bundle_kb=dict(median=float(np.median(sizes)) / 1024, total_mb=sum(sizes) / 2**20),
           hapi_ours=summarise(v_ours), hapi_native=summarise(v_nat), concordance=conc)
json.dump(out, open(f"{OUT}/fhir.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("resources", "ontology_coding", "export_ms_per_patient", "bundle_kb", "hapi_ours", "hapi_native", "concordance")}, indent=1))
