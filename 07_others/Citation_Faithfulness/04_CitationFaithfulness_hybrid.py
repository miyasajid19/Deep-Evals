from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics.community import CitationFaithfulnessMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Citation Faithfulness: the LLM extraction
# step is preserved. The actual judgement — "is the
# answer faithful?" — is delegated to Jev, as a single
# yes/no question, with `P(yes) >= 0.5` counted as
# faithful.
#
# The reason states Jev's probability and confidence.
# If a Jev call fails, the metric silently defers to
# the LLM.

hybrid_citation = CitationFaithfulnessMetric(
    threshold=1.0,
    model=CustomOpenAI(),       # LLM still extracts claim/citation pairs
    include_reason=True,
    eval_mode="hybrid",         # Jev answers faithful / unfaithful
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Both citations correct, all claims supported. Expect
# Jev to answer yes (faithful).

test_case = LLMTestCase(
    input="How tall is the Eiffel Tower and when was it completed?",
    actual_output="The Eiffel Tower stands 330 metres tall [1] and was completed in 1889 [2].",
    retrieval_context=[
        "The Eiffel Tower stands 330 metres tall in Paris.",
        "The Eiffel Tower was completed in 1889 for the World Fair.",
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# `metric.confidence` reports how decisive Jev was.

hybrid_citation.measure(test_case)
print("---")
print(f"Score:      {hybrid_citation.score}")
print(f"Confidence: {hybrid_citation.confidence}")
print(f"Reason:     {hybrid_citation.reason}")
print(f"Passed:     {hybrid_citation.is_successful()}")