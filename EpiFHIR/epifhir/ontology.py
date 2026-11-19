"""Load an OWL epilepsy ontology (EPIO) into a flat lexicon.

Each class contributes its rdfs:label and its exact synonyms as surface forms.
Cross-references (SNOMED CT, ICD-10-CM, ICD-9-CM, UMLS, MeSH) are kept so that
coded EHR data can be mapped to ontology classes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import rdflib
from rdflib.namespace import OWL, RDF, RDFS

OIO = "http://www.geneontology.org/formats/oboInOwl#"
EXACT = rdflib.URIRef(OIO + "hasExactSynonym")
RELATED = rdflib.URIRef(OIO + "hasRelatedSynonym")
XREF = (rdflib.URIRef(OIO + "hasDbXref"), rdflib.URIRef(OIO + "hasDbXRef"))
FROM_EPSO = rdflib.URIRef("https://bio.scai.fraunhofer.de/ontology/epilepsy#fromEpSO")

_PREFIX = {
    "SNOMEDCT_US_2018_03_01": "SNOMED", "SNOMEDCT_US": "SNOMED", "SNOMEDCT": "SNOMED",
    "SCTID": "SNOMED", "ICD10CM": "ICD10CM", "ICD9CM": "ICD9CM", "UMLS": "UMLS",
    "UMLS_CUI": "UMLS", "MESH": "MESH", "MSH": "MESH",
}


@dataclass
class OntClass:
    iri: str
    label: str
    exact: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    xrefs: dict[str, set[str]] = field(default_factory=dict)
    from_epso: bool = False


def _norm_code(system: str, code: str) -> str:
    code = code.strip()
    if system == "ICD10CM" or system == "ICD9CM":
        code = code.replace(".", "").upper()
    return code


def load_ontology(path: str) -> dict[str, OntClass]:
    g = rdflib.Graph()
    g.parse(path)
    out: dict[str, OntClass] = {}
    # sorted() makes iteration order (and therefore tie-breaking) independent of
    # Python's per-process string-hash randomisation
    for s in sorted({x for x in g.subjects(RDF.type, OWL.Class) if isinstance(x, rdflib.URIRef)}, key=str):
        labels = sorted(str(o) for o in g.objects(s, RDFS.label))
        if not labels:
            continue
        c = OntClass(iri=str(s), label=labels[0])
        c.exact = sorted({str(o) for o in g.objects(s, EXACT)} - {c.label})
        c.related = sorted({str(o) for o in g.objects(s, RELATED)})
        c.from_epso = any(True for _ in g.objects(s, FROM_EPSO))
        for p in XREF:
            for o in g.objects(s, p):
                m = re.match(r"^([A-Za-z0-9_]+):(.+)$", str(o).strip())
                if not m:
                    continue
                system = _PREFIX.get(m.group(1).upper(), None) or _PREFIX.get(m.group(1), None)
                if system:
                    c.xrefs.setdefault(system, set()).add(_norm_code(system, m.group(2)))
        out[c.iri] = c
    return out


def xref_index(classes: dict[str, OntClass], system: str) -> dict[str, set[str]]:
    """code -> set of class IRIs."""
    idx: dict[str, set[str]] = {}
    for c in classes.values():
        for code in c.xrefs.get(system, ()):
            idx.setdefault(code, set()).add(c.iri)
    return idx
