"""Cross-implementation concordance (UTC-normalised) and classification of
HAPI messages for Synthea's native bundles."""
import glob, json, os, subprocess
from datetime import datetime, timezone
R = "results"
nat = {os.path.basename(f).rsplit("_", 1)[-1][:-5]: f for f in glob.glob("data/synthea/output/fhir/*.json")}

def inst(s):
    if not s: return ""
    if "T" not in s: return s[:10]
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

def ks(b, t):
    out = set()
    for e in b["entry"]:
        r = e["resource"]
        if r["resourceType"] != t: continue
        if t == "Condition": out.add((r["code"]["coding"][0]["code"], (r.get("onsetDateTime") or "")[:10]))  # CSV writes condition dates in the generator local zone
        elif t == "MedicationRequest": out.add((r["medicationCodeableConcept"]["coding"][0]["code"], inst(r.get("authoredOn"))))
        elif t == "Procedure": out.add((r["code"]["coding"][0]["code"], inst(r.get("performedPeriod", {}).get("start") or r.get("performedDateTime"))))
    return out

conc = {t: dict(ours=0, native=0, both=0, patients_identical=0, patients_compared=0) for t in ("Condition", "MedicationRequest", "Procedure")}
for f in sorted(glob.glob(f"{R}/bundles/*.json")):
    pid = os.path.basename(f)[:-5]
    o, n = json.load(open(f)), json.load(open(nat[pid]))
    for t, c in conc.items():
        a, b = ks(o, t), ks(n, t)
        c["ours"] += len(a); c["native"] += len(b); c["both"] += len(a & b)
        c["patients_identical"] += int(a == b); c["patients_compared"] += 1
for c in conc.values():
    c["jaccard"] = c["both"] / (c["ours"] + c["native"] - c["both"])

# classify native messages on the same 100-bundle sample
sample = [x["file"] for x in json.load(open(f"{R}/hapi_native.json"))]
env = dict(os.environ, ALLMSG="1")
out = subprocess.run(["java", "-Xmx3g", "-cp", "tools/synthea.jar:validator", "Validate", *sample],
                     capture_output=True, text=True, env=env).stdout
cats = {"unknown US Core profile (not available offline)": 0, "other": 0}
other = {}
for line in out.splitlines():
    if not line.startswith("MSG\t"): continue
    m = line.split("\t", 2)[2]
    if "has not been checked because it is unknown" in m: cats["unknown US Core profile (not available offline)"] += 1
    else:
        cats["other"] += 1; key = m.split(" :: ", 1)[-1][:120]; other[key] = other.get(key, 0) + 1
d = json.load(open(f"{R}/fhir.json"))
d["concordance"] = conc
d["hapi_native"]["error_categories"] = cats
d["hapi_native"]["other_top"] = sorted(other.items(), key=lambda x: -x[1])[:8]
json.dump(d, open(f"{R}/fhir.json", "w"), indent=1)
print(json.dumps(conc, indent=1)); print(cats); print(d["hapi_native"]["other_top"])
