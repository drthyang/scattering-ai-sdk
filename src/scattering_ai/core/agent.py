"""Single-agent analysis loop (roadmap A2).

Flow: validate request → deterministic diagnostics → knowledge retrieval →
LLM interpretation → validated report with provenance.

The loop degrades gracefully: with no LLM configured it still produces a
useful deterministic report (findings + rule-based next checks), so the SDK
works offline and tests never need a model.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import scattering_ai
from scattering_ai.core.findings import Finding, Severity
from scattering_ai.core.schemas import (
    AnalysisReport,
    AnalysisRequest,
    Citation,
    Confidence,
    Provenance,
)
from scattering_ai.domains.registry import DomainPack, get_domain
from scattering_ai.llm.base import LLMClient, Message
from scattering_ai.rag.retriever import KnowledgeBase, RetrievedChunk, default_knowledge_root

# Deterministic next-check rules, applied per diagnostic type when no LLM is
# available (and merged in as a floor even when one is).
NEXT_CHECK_RULES: dict[str, str] = {
    "rwp_trend:flat": "Check the move acceptance rate; near-zero acceptance suggests "
    "over-tight constraints, very high acceptance suggests loose dataset weights.",
    "rwp_trend:oscillating": "Inspect dataset weights for competing datasets or "
    "constraints pulling the configuration in opposite directions.",
    "rwp_trend:increasing": "Check for mid-run changes to weights or constraints and "
    "verify the restart configuration file.",
    "dataset_kind_conflict": "Compare partial PDFs and rebalance dataset weights; a "
    "robust local-structure signal should survive moderate weight changes.",
    "radiation_conflict": "Identify which partial correlations dominate each probe "
    "(neutron b vs x-ray Z weighting) and check element-specific misfit.",
    "missing_files": "Locate or regenerate the missing files before trusting the analysis.",
    "log_error": "Inspect the run log around the reported error line.",
    "log_warning": "Review the warning in the run log and confirm it is benign.",
}


def _rule_key(finding: Finding) -> str:
    trend = finding.evidence.get("trend")
    return f"{finding.diagnostic}:{trend}" if trend else finding.diagnostic


def _input_hash(request: AnalysisRequest) -> str:
    canonical = request.model_dump_json()
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _parse_llm_json(content: str) -> dict | None:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


class Agent:
    def __init__(
        self,
        llm: LLMClient | None = None,
        model_id: str = "",
        knowledge_root: Path | str | None = None,
    ):
        self.llm = llm
        self.model_id = model_id
        self.knowledge_root = knowledge_root or default_knowledge_root()

    def analyze(self, request: AnalysisRequest) -> AnalysisReport:
        pack = get_domain(request.domain)
        findings = pack.run_diagnostics(request)
        chunks = self._retrieve(request, pack)

        observations = [f.message for f in findings]
        warnings = [f.message for f in findings if f.severity != Severity.INFO]
        rule_checks = sorted(
            {NEXT_CHECK_RULES[key] for f in findings if (key := _rule_key(f)) in NEXT_CHECK_RULES}
        )

        summary, interpretation, llm_checks, confidence = self._interpret(
            request, pack, findings, chunks
        )
        next_checks = llm_checks + [c for c in rule_checks if c not in llm_checks]

        return AnalysisReport(
            status="ok",
            summary=summary or self._deterministic_summary(findings),
            observations=observations,
            interpretation=interpretation,
            warnings=warnings,
            recommended_next_checks=next_checks,
            citations=[
                Citation(source=r.chunk.path, section=r.chunk.section) for r in chunks
            ],
            confidence=confidence,
            provenance=Provenance(
                sdk_version=scattering_ai.__version__,
                input_hash=_input_hash(request),
                model=self.model_id if self.llm else "",
                prompt_version=pack.prompt_version if self.llm else "",
                retrieved_chunks=[r.citation for r in chunks],
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )

    def _retrieve(self, request: AnalysisRequest, pack: DomainPack) -> list[RetrievedChunk]:
        if not request.options.use_rag or self.knowledge_root is None:
            return []
        kb = KnowledgeBase(self.knowledge_root, subdirs=pack.knowledge_dirs or None)
        return kb.retrieve(request.question, k=4)

    def _deterministic_summary(self, findings: list[Finding]) -> str:
        if not findings:
            return "No diagnostics produced findings; the input may contain too little data."
        n_err = sum(1 for f in findings if f.severity == Severity.ERROR)
        n_warn = sum(1 for f in findings if f.severity == Severity.WARNING)
        if n_err:
            return f"Diagnostics found {n_err} error(s) and {n_warn} warning(s); see evidence."
        if n_warn:
            return f"Diagnostics found {n_warn} warning(s); see evidence and next checks."
        return "All deterministic diagnostics look healthy."

    def _interpret(
        self,
        request: AnalysisRequest,
        pack: DomainPack,
        findings: list[Finding],
        chunks: list[RetrievedChunk],
    ) -> tuple[str, list[str], list[str], Confidence]:
        """Ask the LLM for summary/interpretation; empty results if no LLM."""
        if self.llm is None:
            return "", [], [], Confidence.LOW

        knowledge_block = "\n\n".join(
            f"[K{i + 1}] ({r.citation})\n{r.chunk.text}" for i, r in enumerate(chunks)
        )
        user_content = (
            f"QUESTION:\n{request.question}\n\n"
            f"DIAGNOSTICS (deterministic findings with evidence):\n"
            f"{json.dumps([f.model_dump() for f in findings], indent=1)}\n\n"
            f"KNOWLEDGE:\n{knowledge_block or '(no knowledge retrieved)'}\n\n"
            f"RUN DATA (structured input):\n{request.data.model_dump_json()}"
        )
        response = self.llm.complete(
            [
                Message(role="system", content=pack.system_prompt),
                Message(role="user", content=user_content),
            ]
        )
        parsed = _parse_llm_json(response.content)
        if parsed is None:
            return (
                "",
                ["LLM interpretation unavailable: response was not valid JSON."],
                [],
                Confidence.LOW,
            )
        try:
            confidence = Confidence(parsed.get("confidence", "low"))
        except ValueError:
            confidence = Confidence.LOW
        as_list = lambda v: [str(x) for x in v] if isinstance(v, list) else []  # noqa: E731
        return (
            str(parsed.get("summary", "")),
            as_list(parsed.get("interpretation")),
            as_list(parsed.get("recommended_next_checks")),
            confidence,
        )


def analyze(
    domain: str,
    question: str,
    data: dict | None = None,
    llm: LLMClient | None = None,
    model_id: str = "",
    **options,
) -> AnalysisReport:
    """One-call public API: build a request, run the agent, return the report."""
    request = AnalysisRequest(
        domain=domain,
        question=question,
        data=data or {},
        options=options or {},
    )
    return Agent(llm=llm, model_id=model_id).analyze(request)
