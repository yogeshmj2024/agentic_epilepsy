"""Three-stage concept alignment of clinical text to ontology classes.

Stage 1  normalisation: Unicode folding, lower-casing, punctuation removal,
         stop-word removal and lemmatisation (simplemma).
Stage 2  candidate generation: exact match on the normalised form, otherwise a
         token index with prefix matching (prefix trie over tokens, >= 4 chars).
Stage 3  disambiguation: character 3-5-gram TF-IDF cosine between the query and
         each candidate surface form; the best class is accepted if its score
         is >= tau, otherwise the aligner abstains (manual review).
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

import numpy as np
import simplemma
from sklearn.feature_extraction.text import TfidfVectorizer

STOP = {
    "a", "an", "the", "of", "and", "or", "with", "without", "in", "on", "to", "for",
    "by", "at", "as", "is", "be", "due", "type", "nos", "other", "unspecified",
    "disorder", "finding", "situation", "(", ")",
}
_tok = re.compile(r"[a-z0-9]+")


def normalise(text: str, lemmatise: bool = True) -> tuple[str, ...]:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    toks = [w for w in _tok.findall(t) if w not in STOP]
    if lemmatise:
        toks = [simplemma.lemmatize(w, lang="en") for w in toks]
    return tuple(toks)


class _Trie:
    """Prefix trie over vocabulary tokens -> token ids."""

    def __init__(self):
        self.root: dict = {}

    def add(self, word: str):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node["$"] = word

    def complete(self, prefix: str, limit: int = 50) -> list[str]:
        node = self.root
        for ch in prefix:
            if ch not in node:
                return []
            node = node[ch]
        out, stack = [], [node]
        while stack and len(out) < limit:
            n = stack.pop()
            for k, v in n.items():
                if k == "$":
                    out.append(v)
                else:
                    stack.append(v)
        return out


class Aligner:
    def __init__(self, surface_forms: list[tuple[str, str]], *, lemmatise=True,
                 use_trie=True, use_stage3=True, tau=0.5, min_prefix=4):
        """surface_forms: list of (text, class_iri)."""
        self.lemmatise, self.use_trie, self.use_stage3 = lemmatise, use_trie, use_stage3
        self.tau, self.min_prefix = tau, min_prefix
        self.forms = [(t, iri, normalise(t, lemmatise)) for t, iri in surface_forms]
        self.forms = [f for f in self.forms if f[2]]
        self.exact: dict[tuple, set[str]] = defaultdict(set)
        self.inv: dict[str, set[int]] = defaultdict(set)
        self.trie = _Trie()
        for i, (_, iri, nt) in enumerate(self.forms):
            self.exact[nt].add(iri)
            for w in set(nt):
                self.inv[w].add(i)
        for w in self.inv:
            self.trie.add(w)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.M = self.vec.fit_transform([" ".join(f[2]) for f in self.forms])

    # ------------------------------------------------------------------
    def candidates(self, nq: tuple[str, ...]) -> list[int]:
        if not self.use_trie:
            return list(range(len(self.forms)))
        # candidate order is sorted so that ties resolve identically across runs
        cand: set[int] = set()
        for w in nq:
            if w in self.inv:
                cand |= self.inv[w]
            elif len(w) >= self.min_prefix:
                for full in self.trie.complete(w[: max(self.min_prefix, len(w) - 1)]):
                    cand |= self.inv[full]
        return sorted(cand)

    def score(self, nq: tuple[str, ...], cand: list[int]) -> np.ndarray:
        if self.use_stage3:
            q = self.vec.transform([" ".join(nq)])
            return (self.M[cand] @ q.T).toarray().ravel()
        # Stage 3 ablated: token Jaccard only
        s = set(nq)
        return np.array([len(s & set(self.forms[i][2])) / len(s | set(self.forms[i][2]))
                         for i in cand])

    def align(self, text: str) -> tuple[str | None, float, str]:
        """Return (class_iri or None, score, route)."""
        nq = normalise(text, self.lemmatise)
        if not nq:
            return None, 0.0, "empty"
        hit = self.exact.get(nq)
        if hit and len(hit) == 1:
            return next(iter(hit)), 1.0, "exact"
        cand = self.candidates(nq)
        if not cand:
            return None, 0.0, "no_candidate"
        sc = self.score(nq, cand)
        # deterministic tie-break: highest score, then lowest form index
        best = int(np.argmax(sc))
        if sc[best] >= self.tau:
            return self.forms[cand[best]][1], float(sc[best]), "scored"
        return None, float(sc[best]), "below_tau"


def surface_forms(classes, include_exact=True, drop: dict[str, set[str]] | None = None):
    drop = drop or {}
    out = []
    for c in classes.values():
        forms = [c.label] + (c.exact if include_exact else [])
        for f in forms:
            if f in drop.get(c.iri, ()):
                continue
            out.append((f, c.iri))
    return out
