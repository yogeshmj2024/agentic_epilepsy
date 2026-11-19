"""SQLite vs PostgreSQL behind one SQLAlchemy schema.

The synthetic cohort is replicated k times with fresh identifiers to reach
larger sizes. For each back end and size we measure bulk-load throughput,
an ontology-driven cohort query, per-patient record retrieval, and concurrent
single-row writes from several threads.
"""
import json
import os
import random
import statistics
import sys
import threading
import time
import uuid

import pandas as pd
from sqlalchemy import (Column, Float, ForeignKey, Index, MetaData, String, Table, create_engine,
                        event, func, select, text)

SYN = "data/synthea/output/csv"
OUT = "results"
SEIZ = "128613002"
rng = random.Random(11)

md = MetaData()
patients = Table("patients", md, Column("id", String(40), primary_key=True), Column("gender", String(2)),
                 Column("birthdate", String(10)))
encounters = Table("encounters", md, Column("id", String(40), primary_key=True),
                   Column("patient", String(40), ForeignKey("patients.id"), index=True),
                   Column("start", String(25)), Column("klass", String(20)), Column("code", String(20)))
conditions = Table("conditions", md, Column("id", String(40), primary_key=True),
                   Column("patient", String(40), ForeignKey("patients.id"), index=True),
                   Column("encounter", String(40)), Column("code", String(20), index=True),
                   Column("onset", String(10)), Column("ontology_class", String(120), index=True))
medications = Table("medications", md, Column("id", String(40), primary_key=True),
                    Column("patient", String(40), ForeignKey("patients.id"), index=True),
                    Column("code", String(20)), Column("start", String(25)), Column("cost", Float))


def load_frames():
    rd = lambda n: pd.read_csv(f"{SYN}/{n}.csv", dtype=str)  # noqa: E731
    p, e, c, m = rd("patients"), rd("encounters"), rd("conditions"), rd("medications")
    ont = json.load(open(f"{OUT}/fhir.json"))["ontology_by_condition_code"]
    omap = {k.split("|")[0]: v for k, v in ont.items() if v}
    return p, e, c, m, omap


def replicate(p, e, c, m, omap, k):
    P, E, C, M = [], [], [], []
    for r in range(k):
        pid = {x: str(uuid.uuid5(uuid.NAMESPACE_OID, f"{r}-{x}")) for x in p.Id}
        eid = {x: str(uuid.uuid5(uuid.NAMESPACE_OID, f"{r}-e-{x}")) for x in e.Id}
        P += [dict(id=pid[x], gender=g, birthdate=b) for x, g, b in zip(p.Id, p.GENDER, p.BIRTHDATE)]
        E += [dict(id=eid[x], patient=pid[q], start=s, klass=kl, code=cd)
              for x, q, s, kl, cd in zip(e.Id, e.PATIENT, e.START, e.ENCOUNTERCLASS, e.CODE)]
        C += [dict(id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"{r}-c-{i}")), patient=pid[q], encounter=eid.get(en),
                   code=cd, onset=s, ontology_class=omap.get(cd)) for i, (q, en, cd, s) in
              enumerate(zip(c.PATIENT, c.ENCOUNTER, c.CODE, c.START))]
        M += [dict(id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"{r}-m-{i}")), patient=pid[q], code=cd, start=s,
                   cost=float(tc) if tc == tc else None) for i, (q, cd, s, tc) in
              enumerate(zip(m.PATIENT, m.CODE, m.START, m.TOTALCOST))]
    return P, E, C, M


def engine(kind):
    if kind == "sqlite":
        path = "/tmp/epi_bench.sqlite"
        if os.path.exists(path):
            os.remove(path)
        eng = create_engine(f"sqlite:///{path}", connect_args={"timeout": 30, "check_same_thread": False})

        @event.listens_for(eng, "connect")
        def _pragma(dbapi, _):
            dbapi.execute("PRAGMA journal_mode=WAL")
            dbapi.execute("PRAGMA synchronous=NORMAL")
        return eng
    eng = create_engine("postgresql+psycopg2://epi:epi@localhost/epi", pool_size=8, max_overflow=4)
    with eng.begin() as cx:
        cx.execute(text("DROP TABLE IF EXISTS medications, conditions, encounters, patients CASCADE"))
    return eng


def bench(kind, P, E, C, M, threads=4, writes_per_thread=500):
    eng = engine(kind)
    md.create_all(eng)
    t0 = time.perf_counter()
    with eng.begin() as cx:
        for tbl, rows in ((patients, P), (encounters, E), (conditions, C), (medications, M)):
            for i in range(0, len(rows), 5000):
                cx.execute(tbl.insert(), rows[i:i + 5000])
    load_s = time.perf_counter() - t0
    nrows = len(P) + len(E) + len(C) + len(M)

    q = (select(conditions.c.ontology_class, func.count(func.distinct(conditions.c.patient)))
         .where(conditions.c.ontology_class.is_not(None)).group_by(conditions.c.ontology_class))
    lat = []
    for _ in range(5):
        t1 = time.perf_counter()
        with eng.connect() as cx:
            res = cx.execute(q).fetchall()
        lat.append(1000 * (time.perf_counter() - t1))
    cohort_ms = statistics.median(lat)

    ids = [p["id"] for p in rng.sample(P, min(300, len(P)))]
    rec = []
    with eng.connect() as cx:
        for pid in ids:
            t1 = time.perf_counter()
            for tbl in (encounters, conditions, medications):
                cx.execute(select(tbl).where(tbl.c.patient == pid)).fetchall()
            rec.append(1000 * (time.perf_counter() - t1))
    rec.sort()

    errors, done = [0], [0]
    lock = threading.Lock()

    def writer(tid):
        for j in range(writes_per_thread):
            try:
                with eng.begin() as cx:
                    cx.execute(conditions.insert(), dict(id=f"w{tid}-{j}-{uuid.uuid4().hex[:24]}", patient=P[j % len(P)]["id"],
                                                         encounter=None, code=SEIZ, onset="2026-01-01",
                                                         ontology_class=None))
                with lock:
                    done[0] += 1
            except Exception:
                with lock:
                    errors[0] += 1
    t1 = time.perf_counter()
    th = [threading.Thread(target=writer, args=(i,)) for i in range(threads)]
    [t.start() for t in th]
    [t.join() for t in th]
    wr_s = time.perf_counter() - t1
    eng.dispose()
    return dict(backend=kind, patients=len(P), rows=nrows, load_s=load_s, load_rows_per_s=nrows / load_s,
                cohort_query_ms=cohort_ms, cohort_classes=len(res),
                record_ms_p50=rec[len(rec) // 2], record_ms_p95=rec[int(0.95 * len(rec))],
                concurrent_threads=threads, concurrent_commits=done[0], concurrent_errors=errors[0],
                concurrent_commits_per_s=done[0] / wr_s)


if __name__ == "__main__":
    p, e, c, m, omap = load_frames()
    rows = []
    for k in (1, 5, 20):
        P, E, C, M = replicate(p, e, c, m, omap, k)
        for kind in ("sqlite", "postgresql"):
            r = bench(kind, P, E, C, M)
            r["replication"] = k
            rows.append(r)
            print(json.dumps(r))
    json.dump(rows, open(f"{OUT}/storage.json", "w"), indent=1)
