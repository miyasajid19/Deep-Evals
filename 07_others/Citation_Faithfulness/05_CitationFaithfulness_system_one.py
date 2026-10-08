from deepeval.test_case import LLMTestCase
from deepeval.metrics.community import CitationFaithfulnessMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# Under `system_one` mode, Jev judges the whole metric
# in one request. No LLM is called at all. Jev is sent
# the `actual_output` and the numbered passages and
# asked three weighted questions:
#
#   Q1 (Noul, weight 2): every [N] citation marker in
#                        actual_output is attached to
#                        a claim that passage [N] in
#                        passages supports.
#   Q2 (Noul, weight 1): every factual claim in
#                        actual_output is supported by
#                        passages, and none contradicts
#                        them.
#   Q3 (Score, weight 1): how many of the [N] citation
#                        markers in actual_output point
#                        to a passage in passages that
#                        supports the claim they are
#                        attached to? (None of them
#                        → All of them)
#
# Each answer becomes a value in [0, 1] and the score is
# their weighted mean. The metric is pass/fail at the
# end: weighted mean >= 0.5 scores 1, anything lower
# scores 0.
#
# NOTE: a system_one score comes from Jev's answers, not
# from the LLM chain. Don't compare across eval modes.

system_one_citation = CitationFaithfulnessMetric(
    threshold=1.0,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# All citations correct, all claims supported.

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

system_one_citation.measure(test_case)
print("---")
print(f"Score:      {system_one_citation.score}")
print(f"Confidence: {system_one_citation.confidence}")
print(f"Reason:     {system_one_citation.reason}")
print(f"Passed:     {system_one_citation.is_successful()}")