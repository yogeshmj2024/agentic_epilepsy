# EpiFHIR

EpiFHIR maps epilepsy concepts to the Epilepsy Ontology (EPIO), exports patient records as validated HL7 FHIR R4 bundles, and stores them in SQLite or PostgreSQL for analysis. This repository contains the code, benchmark scripts and result files behind the paper *"EpiFHIR: an open, reproducible pipeline that links epilepsy records to an ontology and HL7 FHIR R4, evaluated end to end on public synthetic patients"*.

Every input is public, so every number in the paper can be regenerated.

## Layout

| Path | Contents |
|---|---|
| `epifhir/ontology.py` | Loads an OWL ontology: labels, synonyms and cross-references |
| `epifhir/align.py` | Three-stage aligner (normalise and lemmatise, prefix-trie candidates, character n-gram scoring with abstention) |
| `epifhir/fhir_export.py` | FHIR R4 exporter; `coded_text_policy="review"` (default) sends un-cross-referenced coded concepts to a review queue |
| `validator/Validate.java` | Offline HAPI FHIR R4 validator (uses the HAPI classes bundled in the Synthea jar) |
| `bench/` | One script per experiment, plus `make_tables.py` and `figures.py` |
| `synthea/` | Synthea settings (`synthea.properties`) and the keep-module filter (`keep.json`) |
| `results/` | Result files used in the paper (JSON/NPY/TSV) |

## Reproduce

1. **Requirements:** Python 3.11 or later, Java 17 or later, and PostgreSQL 16 (for the storage benchmark; a user `epi` with password `epi` and a database `epi` on localhost). Then run `pip install -r requirements.txt`.
2. **Tools and data:**
   ```bash
   mkdir -p tools data
   curl -L -o tools/synthea.jar https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
   git clone https://github.com/SCAI-BIO/EpilepsyOntology data/EpilepsyOntology   # paper used commit dc9f9b1
   cd data && mkdir synthea && cd synthea
   java -jar ../../tools/synthea.jar -s 42 -cs 42 -p 1500 -m epilepsy \
        -k ../../synthea/keep.json -c ../../synthea/synthea.properties Massachusetts
   cd ../..
   ```
   The paper used Synthea build `bd8a52b`. Newer builds may produce a slightly different cohort.
3. **Run:** `./run_all.sh`. On a two-core machine this takes about 20 minutes plus generation time.

Timing results (milliseconds, rows per second) depend on hardware. All other results are deterministic given the seeds.

## Licence notes

EPIO is MIT-licensed (Fraunhofer SCAI). Synthea is Apache-2.0. ICD-10-CM titles come from the `simple-icd-10-cm` package.
