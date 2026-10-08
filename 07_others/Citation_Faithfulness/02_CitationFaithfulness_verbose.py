from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics.community import CitationFaithfulnessMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the per-citation verdict:
# which `[N]` was checked, what the claim was, whether
# passage [N] supports it, and the final faithful /
# unfaithful classification. The fastest way to debug
# "the answer is right but cited the wrong passage".

citation_faithfulness = CitationFaithfulnessMetric(
    threshold=1.0,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — perfectly cited run
# ==========================================
# Height claim cites [1] (the height passage) and
# completion-year claim cites [2] (the completion
# passage). Both citations correctly point to passages
# that support them. Expect score 1.0.

test_case = LLMTestCase(
    input="How tall is the Eiffel Tower and when was it completed?",
    actual_output="The Eiffel Tower was completed in 1889 [2] and stands 330 metres tall [1].",
    retrieval_context=[
        "The Eiffel Tower stands 330 metres tall in Paris.",
        "The Eiffel Tower was completed in 1889 for the World Fair.",
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason.
# Verbose output prints the per-citation verdicts
# above.

citation_faithfulness.measure(test_case)
print("---")
print(f"Score:   {citation_faithfulness.score}")
print(f"Reason:  {citation_faithfulness.reason}")
print(f"Passed:  {citation_faithfulness.is_successful()}")