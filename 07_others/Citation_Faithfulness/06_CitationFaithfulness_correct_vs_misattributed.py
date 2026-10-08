from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics.community import CitationFaithfulnessMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# Demo comparing correctly-cited vs misattributed
# responses. Same factual claim, different citation
# marker — CitationFaithfulness flags the misattribution
# that a regular FaithfulnessMetric would let through.

citation_faithfulness = CitationFaithfulnessMetric(
    threshold=1.0,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED RETRIEVAL CONTEXT
# ==========================================
# Two passages, both relevant.
#
#   [1] height: 330 metres tall
#   [2] completion: 1889 for the World Fair

retrieval_context = [
    "The Eiffel Tower stands 330 metres tall in Paris.",
    "The Eiffel Tower was completed in 1889 for the World Fair.",
]

# ==========================================
# 3. CASE A — correctly cited
# ==========================================
# Completion-year claim cites [2], which IS the
# completion passage. Pass.

correctly_cited = LLMTestCase(
    input="When was the Eiffel Tower completed?",
    actual_output="The Eiffel Tower was completed in 1889 [2].",
    retrieval_context=retrieval_context,
)

# ==========================================
# 4. CASE B — misattributed citation
# ==========================================
# Same factual claim ("completed in 1889"), but cites
# [1] (the height passage). The claim is TRUE and is
# supported by something in the context, but the cited
# passage doesn't actually support it. Fail.

misattributed = LLMTestCase(
    input="When was the Eiffel Tower completed?",
    actual_output="The Eiffel Tower was completed in 1889 [1].",
    retrieval_context=retrieval_context,
)

# ==========================================
# 5. RUN THE METRIC ON BOTH CASES
# ==========================================

def run(label, case):
    citation_faithfulness.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {citation_faithfulness.score:.3f}")
    print(f"  Passed: {citation_faithfulness.is_successful()}")
    print(f"  Reason: {citation_faithfulness.reason}")

run("A — correctly cited [2]",  correctly_cited)
run("B — misattributed to [1]", misattributed)