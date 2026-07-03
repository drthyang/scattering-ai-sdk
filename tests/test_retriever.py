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
