"""Keyword retrieval over the curated Markdown knowledge base.

Deliberately simple (roadmap B2): chunks are ``##`` sections of Markdown
files, scored by TF-IDF-weighted term overlap with the query. Citations are
(file path, section title) pairs. An embedding backend can replace the
scoring later without changing the ``retrieve`` interface.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from pydantic import BaseModel

_STOPWORDS = frozenset(
    """a an and are as at be but by can do does for from get has have how in is it
    its may might of on or should that the this to was what when where which while
    why will with would""".split()
)

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")


def _tokenize(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]


class Chunk(BaseModel):
    path: str  # relative to the knowledge root
    section: str
    text: str


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float

    @property
    def citation(self) -> str:
        return f"{self.chunk.path}#{self.chunk.section}" if self.chunk.section else self.chunk.path


def default_knowledge_root() -> Path | None:
    """Locate the knowledge/ directory for both installed and repo layouts."""
    package_dir = Path(__file__).resolve().parents[1]
    candidates = [
        package_dir / "knowledge",  # bundled into the wheel
        package_dir.parents[1] / "knowledge",  # src layout: repo root
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split Markdown into (section_title, body) pairs on ## headings."""
    sections: list[tuple[str, str]] = []
    title = ""
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if lines:
                sections.append((title, "\n".join(lines).strip()))
            title = line[3:].strip()
            lines = []
        elif line.startswith("# "):
            continue  # file title: contributes via path, not as a chunk
        else:
            lines.append(line)
    if lines:
        sections.append((title, "\n".join(lines).strip()))
    return [(t, b) for t, b in sections if b]


class KnowledgeBase:
    def __init__(self, root: Path | str, subdirs: list[str] | None = None):
        self.root = Path(root)
        self.chunks: list[Chunk] = []
        self._doc_freq: dict[str, int] = {}
        self._load(subdirs)

    def _load(self, subdirs: list[str] | None) -> None:
        roots = [self.root / d for d in subdirs] if subdirs else [self.root]
        for base in roots:
            for path in sorted(base.rglob("*.md")):
                rel = str(path.relative_to(self.root))
                for section, body in _split_sections(path.read_text()):
                    self.chunks.append(Chunk(path=rel, section=section, text=body))
        for chunk in self.chunks:
            for term in set(_tokenize(f"{chunk.path} {chunk.section} {chunk.text}")):
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        query_terms = set(_tokenize(query))
        if not query_terms or not self.chunks:
            return []
        n_chunks = len(self.chunks)
        scored: list[RetrievedChunk] = []
        for chunk in self.chunks:
            # Section titles and file names are the strongest relevance signal
            # in a curated knowledge base, so weight them above body text.
            title_terms = _tokenize(f"{chunk.path} {chunk.section}")
            body_terms = _tokenize(chunk.text)
            counts: dict[str, float] = {}
            for term in body_terms:
                counts[term] = counts.get(term, 0) + 1.0
            for term in title_terms:
                counts[term] = counts.get(term, 0) + 5.0
            score = 0.0
            for term in query_terms:
                if term in counts:
                    idf = math.log(1 + n_chunks / self._doc_freq.get(term, 1))
                    score += (1 + math.log(counts[term])) * idf
            if score > 0:
                scored.append(RetrievedChunk(chunk=chunk, score=round(score, 4)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]
