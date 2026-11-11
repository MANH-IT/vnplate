from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


@dataclass
class Observation:
    text: str
    confidence: float
    frame: int | None = None


@dataclass
class FusedResult:
    track_id: int
    text: str
    confidence: float
    observations: int
    support: float
    method: str

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "text": self.text,
            "confidence": self.confidence,
            "observations": self.observations,
            "support": self.support,
            "method": self.method,
        }


class TemporalFusion:
    def __init__(
        self,
        strategy: str = "weighted",
        char_vote: bool = True,
        min_votes: int = 2,
        min_confidence: float = 0.3,
    ):
        self.strategy = strategy
        self.char_vote = char_vote
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self.records: dict[int, list[Observation]] = defaultdict(list)

    def reset(self) -> None:
        self.records.clear()

    def add(self, track_id: int, text: str, confidence: float, frame: int | None = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        self.records[int(track_id)].append(Observation(text=text, confidence=float(confidence), frame=frame))

    def _weighted_string_vote(self, observations: list[Observation]) -> tuple[str, float]:
        weights: Counter[str] = Counter()
        total = 0.0
        for obs in observations:
            weights[obs.text] += obs.confidence
            total += obs.confidence
        if not weights:
            return "", 0.0
        text, weight = weights.most_common(1)[0]
        return text, weight / max(total, 1e-6)

    def _char_vote(self, observations: list[Observation]) -> tuple[str, float]:
        by_len: dict[int, float] = defaultdict(float)
        for obs in observations:
            by_len[len(obs.text)] += obs.confidence
        if not by_len:
            return "", 0.0
        best_len = max(by_len, key=by_len.get)
        group = [o for o in observations if len(o.text) == best_len]
        chars = []
        support = 0.0
        total = sum(o.confidence for o in group) or 1e-6
        for pos in range(best_len):
            votes: Counter[str] = Counter()
            for obs in group:
                votes[obs.text[pos]] += obs.confidence
            char, weight = votes.most_common(1)[0]
            chars.append(char)
            support += weight
        return "".join(chars), support / (total * best_len)

    def fuse(self, track_id: int) -> FusedResult | None:
        observations = self.records.get(int(track_id), [])
        if not observations:
            return None
        eligible = [o for o in observations if o.confidence >= self.min_confidence]
        pool = eligible if eligible else observations
        str_text, str_support = self._weighted_string_vote(pool)
        char_text, char_support = self._char_vote(pool) if self.char_vote else ("", 0.0)
        if self.char_vote and char_text and char_support >= str_support:
            text, support, method = char_text, char_support, "char_vote"
        else:
            text, support, method = str_text, str_support, "weighted_vote"
        mean_conf = sum(o.confidence for o in pool) / len(pool)
        confidence = 0.5 * mean_conf + 0.5 * support
        return FusedResult(
            track_id=int(track_id),
            text=text,
            confidence=float(confidence),
            observations=len(observations),
            support=float(support),
            method=method,
        )

    def results(self, confirmed_only: bool = True) -> list[FusedResult]:
        out = []
        for track_id in self.records:
            result = self.fuse(track_id)
            if result is None:
                continue
            if confirmed_only and result.observations < self.min_votes:
                continue
            out.append(result)
        return sorted(out, key=lambda r: (-r.confidence, r.track_id))

    def best(self) -> FusedResult | None:
        results = self.results(confirmed_only=False)
        return results[0] if results else None
