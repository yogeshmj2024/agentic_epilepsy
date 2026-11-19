#!/usr/bin/env bash
# Reproduce every result, table and figure in the EpiFHIR paper. Run from the repository root.
set -euo pipefail
mkdir -p results figures paper/tables
javac -proc:none -cp tools/synthea.jar validator/Validate.java
python3 bench/bench_align.py   | tee results/bench_align.log
python3 bench/bench_fhir.py    > results/bench_fhir.log
python3 bench/concordance.py   | tee results/concordance.log
python3 bench/bench_storage.py | tee results/bench_storage.log
python3 bench/bench_predict.py | tee results/bench_predict.log
java -cp tools/synthea.jar:validator Validate results/sanity > results/sanity_validation.tsv
python3 bench/make_tables.py
python3 bench/figures.py
echo "All results regenerated."
