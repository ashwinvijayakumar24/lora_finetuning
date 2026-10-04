"""R3 retrieval: find the k train plays most similar to a query play.

The goal is examples with the same *shape* as the query (a strip-sack, a reversed
catch, a two-point run), not the same players. So by default descriptions are
normalized before indexing: player tokens become PLAYER, team-yardline spots become
SPOT, and digits become #. TF-IDF over word 1-2 grams of that normalized text, with
cosine similarity, then ranks train plays. scikit-learn is already installed, so no
new dependency is needed.

Examples are returned least-similar first, so the most similar example sits right
before the query in the prompt.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_PLAYER = re.compile(r"\b(?:[A-Z]{2,3}-)?\d{1,2}-[A-Z][A-Za-z']*\. ?[A-Za-z][A-Za-z'\-]*")
_SPOT = re.compile(r"\b[A-Z]{2,3} \d{1,2}\b")
_CLOCK = re.compile(r"^\(\d{0,2}:\d{2}\)\s*")
_DIGITS = re.compile(r"\d+")


def normalize_desc(desc: str) -> str:
    s = _CLOCK.sub("", desc or "")
    s = _PLAYER.sub(" PLAYER ", s)
    s = _SPOT.sub(" SPOT ", s)
    s = _DIGITS.sub("#", s)
    return s


class TfidfRetriever:
    def __init__(
        self,
        records: Sequence[dict],
        *,
        k: int = 8,
        normalize: bool = True,
        source_id: str = "in-memory",
        exclude_same_game: bool = True,
    ):
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not records:
            raise ValueError("retrieval index needs at least one train record")
        self.records = list(records)
        self.k = k
        self.normalize = normalize
        self.source_id = source_id
        self.exclude_same_game = exclude_same_game
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=1, lowercase=False, token_pattern=r"[A-Za-z#]+"
        )
        self.matrix = self.vectorizer.fit_transform([self._prep(r["desc"]) for r in self.records])

    @classmethod
    def from_jsonl(cls, path: str | Path, *, max_records: int | None = None, seed: int = 0, **kw):
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        if max_records is not None and len(recs) > max_records:
            idx = np.random.default_rng(seed).choice(len(recs), size=max_records, replace=False)
            recs = [recs[i] for i in sorted(idx)]
        h = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
        return cls(recs, source_id=f"{Path(path).name}:{h}:n={len(recs)}", **kw)

    def _prep(self, desc: str) -> str:
        return normalize_desc(desc) if self.normalize else (desc or "")

    def config_id(self) -> str:
        return f"tfidf(k={self.k},norm={self.normalize},src={self.source_id})"

    def query(self, records: Sequence[dict]) -> list[list[int]]:
        """Indices of the top-k train records for each query, least similar first."""
        q = self.vectorizer.transform([self._prep(r["desc"]) for r in records])
        sims = (q @ self.matrix.T).toarray()  # rows are L2-normalized, so this is cosine
        return [self._topk(sims[i], rec) for i, rec in enumerate(records)]

    def _topk(self, s: np.ndarray, rec: dict) -> list[int]:
        s = s.copy()
        # Never retrieve the query itself: by default exclude its whole game, which
        # also keeps same-game plays (same players, same day) from leaking in.
        if self.exclude_same_game and "game_id" in rec:
            s[self._game_mask(rec["game_id"])] = -np.inf
        elif "game_id" in rec and "play_id" in rec:
            same = self._game_mask(rec["game_id"]) & (self._plays == str(rec["play_id"]))
            s[same] = -np.inf
        k = min(self.k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.lexsort((top, -s[top]))]  # most similar first, ties by index
        return [int(t) for t in top[::-1]]  # least similar first

    def _game_mask(self, game_id) -> np.ndarray:
        if not hasattr(self, "_games"):
            self._games = np.array([str(r.get("game_id")) for r in self.records])
            self._plays = np.array([str(r.get("play_id")) for r in self.records])
        return self._games == str(game_id)

    def examples_for(self, record: dict) -> list[dict]:
        idx = self.query([record])[0]
        return [
            {"posteam": self.records[i].get("posteam"), "desc": self.records[i]["desc"], "label": self.records[i]["label"]}
            for i in idx
        ]
