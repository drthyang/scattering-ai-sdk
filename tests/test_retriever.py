from scattering_ai.rag.retriever import KnowledgeBase, default_knowledge_root


def knowledge_base():
    root = default_knowledge_root()
    assert root is not None, "knowledge/ directory not found"
    return KnowledgeBase(root)


def test_knowledge_base_loads_chunks():
    kb = knowledge_base()
    assert len(kb.chunks) > 10
    assert all(chunk.section for chunk in kb.chunks[:5])


def test_pdf_vs_bragg_question_retrieves_local_vs_average():
    kb = knowledge_base()
    results = kb.retrieve("Why can the PDF improve while the Bragg fit gets worse?", k=3)
    assert results
    citations = " ".join(r.citation for r in results)
    assert "bragg_vs_total_scattering.md" in citations


def test_convergence_question_retrieves_convergence_note():
    kb = knowledge_base()
    results = kb.retrieve("Why is this run not improving? R-values are flat.", k=3)
    citations = " ".join(r.citation for r in results)
    assert "convergence_interpretation.md" in citations


def test_citations_have_path_and_section():
    kb = knowledge_base()
    top = kb.retrieve("neutron versus x-ray contrast for light elements", k=1)[0]
    assert "#" in top.citation
    assert top.citation.startswith("scattering/neutron_vs_xray_contrast.md")


def test_domain_subdir_filter():
    root = default_knowledge_root()
    kb = KnowledgeBase(root, subdirs=["scattering"])
    assert all(c.path.startswith("scattering/") for c in kb.chunks)


# --- E9: applied knowledge/learned/ snippets are retrievable ------------------

def test_learned_subdir_is_retrieved_and_missing_is_safe(tmp_path):
    root = tmp_path / "knowledge"
    (root / "scattering").mkdir(parents=True)
    (root / "scattering" / "base.md").write_text(
        "# Base\n\n## Contrast\n\nNeutron and x-ray contrast differ.\n")

    # a missing learned/ dir must not break loading
    kb = KnowledgeBase(root, subdirs=["scattering", "learned"])
    assert kb.chunks and all(c.path.startswith("scattering/") for c in kb.chunks)

    # once a snippet is applied (E9), it is loaded and retrievable
    (root / "learned").mkdir()
    (root / "learned" / "pdf__low_r_artifact.md").write_text(
        "# Learned: low_r_artifact (pdf)\n\n## Low-r ripple\n\n"
        "Below the first bond distance the wiggle is termination ripple, "
        "not a real coordination shell.\n")
    kb2 = KnowledgeBase(root, subdirs=["scattering", "learned"])
    top = kb2.retrieve("is the low-r ripple real coordination or termination?", k=1)
    assert top and top[0].chunk.path.startswith("learned/")


def test_agent_retrieval_includes_learned(tmp_path):
    # the agent wires knowledge/learned/ into every pack's retrieval scope
    import numpy as np

    from scattering_ai import Agent, AnalysisRequest

    root = tmp_path / "knowledge"
    (root / "scattering").mkdir(parents=True)
    (root / "scattering" / "x.md").write_text("# X\n\n## Y\n\nsome text\n")
    (root / "learned").mkdir()
    (root / "learned" / "data__transition.md").write_text(
        "# Learned: transition (data)\n\n## GaNb4Se8 transitions\n\n"
        "GaNb4Se8 shows transitions near 50 K and 29 K, not 39 K.\n")

    x = np.linspace(0, 10, 400)
    p = tmp_path / "scan_T_base_5.0K.dat"
    np.savetxt(p, np.column_stack([x, 1 + np.exp(-((x - 4) ** 2))]))
    agent = Agent(knowledge_root=root)
    report = agent.analyze(AnalysisRequest(
        question="what are the GaNb4Se8 transitions?", data={"files": [str(p)]}))
    assert any("learned/" in c.source for c in report.citations)
