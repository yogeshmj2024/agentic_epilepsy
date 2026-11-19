"""Serialise a flat EHR extract (Synthea CSV layout) to HL7 FHIR R4 bundles,
adding ontology codings to Condition resources.

Resources produced per patient: Patient, Encounter, Condition, Procedure,
MedicationRequest. Ontology codings are attached in two ways:
  1. SNOMED CT -> ontology class through the ontology's own cross-references;
  2. otherwise, text alignment of the condition's display name (Aligner).
"""
from __future__ import annotations

import json
import re
import uuid

import pandas as pd

SNOMED = "http://snomed.info/sct"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
ONT_SYSTEM = "https://bio.scai.fraunhofer.de/ontology/epilepsy"
NS = uuid.UUID("6f1c2a52-1b7e-4c55-9a43-6a1f0f3b9e01")

_ENC_CLASS = {"ambulatory": "AMB", "wellness": "AMB", "outpatient": "AMB", "urgentcare": "AMB",
              "emergency": "EMER", "inpatient": "IMP", "home": "HH", "virtual": "VR", "hospice": "HH",
              "snf": "IMP"}


def _uuid(*parts) -> str:
    return str(uuid.uuid5(NS, "|".join(map(str, parts))))


def _dt(x):
    if pd.isna(x) or x == "":
        return None
    s = str(x)
    return s if "T" in s else s[:10]


def _clean_display(d: str) -> str:
    return re.sub(r"\s*\((disorder|finding|situation|procedure|morphologic abnormality)\)\s*$", "", d)


def _ont_iri_code(iri: str) -> str:
    return iri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


class Exporter:
    def __init__(self, classes, snomed_index, aligner=None, tau=0.65, coded_text_policy="review"):
        """coded_text_policy: what to do when a *coded* concept has no ontology
        cross-reference. "review" (default) leaves it uncoded and records it in
        self.review_queue for a person to map once; "auto" applies the text
        aligner directly (the behaviour evaluated in the paper)."""
        self.classes, self.snomed_index, self.aligner, self.tau = classes, snomed_index, aligner, tau
        self.policy = coded_text_policy
        self.review_queue: dict[tuple[str, str], tuple[str | None, float]] = {}
        self.stats = dict(conditions=0, ont_by_xref=0, ont_by_text=0, ont_none=0, sent_to_review=0)

    def ontology_coding(self, snomed_code: str, display: str):
        iris = self.snomed_index.get(str(snomed_code))
        if iris:
            iri = sorted(iris)[0]
            self.stats["ont_by_xref"] += 1
            return iri, "xref"
        if self.aligner is not None:
            iri, s, _ = self.aligner.align(_clean_display(display))
            if self.policy == "review":
                self.review_queue[(str(snomed_code), display)] = (iri, s)
                self.stats["sent_to_review"] += 1
                return None, None
            if iri is not None and s >= self.tau:
                self.stats["ont_by_text"] += 1
                return iri, "text"
        self.stats["ont_none"] += 1
        return None, None

    def bundle(self, pid: str, pat: pd.Series, enc: pd.DataFrame, cond: pd.DataFrame,
               proc: pd.DataFrame, med: pd.DataFrame) -> dict:
        pref = f"urn:uuid:{pid}"
        entries = []
        p = {"resourceType": "Patient", "id": pid,
             "identifier": [{"system": "https://github.com/synthetichealth/synthea", "value": pid}],
             "name": [{"use": "official", "family": str(pat.LAST), "given": [str(pat.FIRST)]}],
             "gender": {"M": "male", "F": "female"}.get(pat.GENDER, "unknown"),
             "birthDate": _dt(pat.BIRTHDATE)}
        if isinstance(pat.DEATHDATE, str) and pat.DEATHDATE:
            p["deceasedDateTime"] = _dt(pat.DEATHDATE)
        entries.append(p)
        enc_ids = set(cond.ENCOUNTER) | set(proc.ENCOUNTER) | set(med.ENCOUNTER)
        for e in enc[enc.Id.isin(enc_ids)].itertuples(index=False):
            r = {"resourceType": "Encounter", "id": e.Id, "status": "finished",
                 "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                           "code": _ENC_CLASS.get(e.ENCOUNTERCLASS, "AMB")},
                 "type": [{"coding": [{"system": SNOMED, "code": str(e.CODE), "display": e.DESCRIPTION}]}],
                 "subject": {"reference": pref},
                 "period": {"start": _dt(e.START)}}
            if _dt(e.STOP):
                r["period"]["end"] = _dt(e.STOP)
            entries.append(r)
        for i, c in enumerate(cond.itertuples(index=False)):
            self.stats["conditions"] += 1
            coding = [{"system": SNOMED, "code": str(c.CODE), "display": c.DESCRIPTION}]
            iri, how = self.ontology_coding(c.CODE, c.DESCRIPTION)
            if iri:
                coding.append({"system": ONT_SYSTEM, "code": _ont_iri_code(iri),
                               "display": self.classes[iri].label, "userSelected": False})
            r = {"resourceType": "Condition", "id": _uuid(pid, "cond", c.CODE, c.START, i),
                 "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                                "code": "resolved" if _dt(c.STOP) else "active"}]},
                 "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                                                    "code": "confirmed"}]},
                 "code": {"coding": coding, "text": c.DESCRIPTION},
                 "subject": {"reference": pref},
                 "encounter": {"reference": f"urn:uuid:{c.ENCOUNTER}"},
                 "onsetDateTime": _dt(c.START)}
            if _dt(c.STOP):
                r["abatementDateTime"] = _dt(c.STOP)
            entries.append(r)
        for i, pr in enumerate(proc.itertuples(index=False)):
            entries.append({"resourceType": "Procedure", "id": _uuid(pid, "proc", pr.CODE, pr.START, i),
                            "status": "completed",
                            "code": {"coding": [{"system": SNOMED, "code": str(pr.CODE), "display": pr.DESCRIPTION}],
                                     "text": pr.DESCRIPTION},
                            "subject": {"reference": pref},
                            "encounter": {"reference": f"urn:uuid:{pr.ENCOUNTER}"},
                            "performedPeriod": {k: v for k, v in (("start", _dt(pr.START)), ("end", _dt(pr.STOP))) if v}})
        for i, m in enumerate(med.itertuples(index=False)):
            entries.append({"resourceType": "MedicationRequest", "id": _uuid(pid, "med", m.CODE, m.START, i),
                            "status": "stopped" if _dt(m.STOP) else "active", "intent": "order",
                            "medicationCodeableConcept": {"coding": [{"system": RXNORM, "code": str(m.CODE),
                                                                      "display": m.DESCRIPTION}],
                                                          "text": m.DESCRIPTION},
                            "subject": {"reference": pref},
                            "encounter": {"reference": f"urn:uuid:{m.ENCOUNTER}"},
                            "authoredOn": _dt(m.START)})
        return {"resourceType": "Bundle", "id": _uuid(pid, "bundle"), "type": "collection",
                "entry": [{"fullUrl": f"urn:uuid:{r['id']}", "resource": r} for r in entries]}


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
