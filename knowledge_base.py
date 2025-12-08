import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from rapidfuzz import fuzz, process


@dataclass
class Correction:
    original: str
    candidate: str
    score: float


class KnowledgeBase:
    def __init__(self, dict_path: Path, min_score: int = 85) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.dict_path = dict_path
        self.min_score = min_score
        self.terms: List[str] = []
        self._load_terms()

    def _load_terms(self) -> None:
        if not self.dict_path.exists():
            self.logger.warning("Dictionary file %s not found", self.dict_path)
            return
        try:
            with self.dict_path.open("r", encoding="utf-8", errors="ignore") as f:
                self.terms = [line.strip() for line in f if line.strip()]
            self.logger.info("Loaded %d terms from knowledge base", len(self.terms))
        except Exception:
            self.logger.exception("Failed to load knowledge base from %s", self.dict_path)
            self.terms = []

    def _extract_tokens(self, text: str) -> List[str]:
        # Capture words and CJK characters; fall back to whitespace splitting.
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text)
        return tokens if tokens else text.split()

    def search_and_correct(self, text: str, max_candidates: int = 3) -> List[Correction]:
        if not self.terms or not text:
            return []
        corrections: List[Correction] = []
        tokens = self._extract_tokens(text)
        unique_tokens = list(dict.fromkeys(tokens))

        # For each token, find the closest term if it looks misspelled.
        for token in unique_tokens:
            if len(token) < 2:
                continue
            matches = process.extract(
                token,
                self.terms,
                scorer=fuzz.QRatio,
                limit=max_candidates,
            )
            for candidate, score, _ in matches:
                if score >= self.min_score and candidate != token:
                    corrections.append(Correction(original=token, candidate=candidate, score=float(score)))
        return corrections

    def build_hint(self, corrections: Iterable[Correction]) -> str:
        items = list(corrections)
        if not items:
            return ""
        parts = [f"{c.original}->{c.candidate} (score={c.score:.1f})" for c in items]
        return "Potential legal term corrections: " + "; ".join(parts)
