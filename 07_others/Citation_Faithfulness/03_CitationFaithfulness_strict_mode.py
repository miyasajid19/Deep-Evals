from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics.community import CitationFaithfulnessMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` enforces a binary metric score:
#   - 0 for perfection (faithful = pass)
#   - 1 otherwise (any unfaithful citation = fail)
# It also overrides `threshold` to 1. Note this metric
# is INVERTED compared to most: the LLM default already
# requires a faithful verdict (threshold=1.0), so strict
# just locks the binary interpretation.
#
# Use it as a CI gate: any misattribution, unsupported
# claim, or contradiction is an unconditional fail.

strict_citation = CitationFaithfulnessMetric(
    threshold=1.0,
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. FAITHFUL CITATIONS — pass (score 0)
# ==========================================
# Height cites [1], completion cites [2]. Expect 0
# (pass).

faithful_case = LLMTestCase(
    input="How tall is the Eiffel Tower and when was it completed?",
    actual_output="The Eiffel Tower stands 330 metres tall [1] and was completed in 1889 [2].",
    retrieval_context=[
        "The Eiffel Tower stands 330 metres tall in Paris.",
        "The Eiffel Tower was completed in 1889 for the World Fair.",
    ],
)

strict_citation.measure(faithful_case)
print("--- all citations faithful ---")
print(f"Score:   {strict_citation.score}")          # 0 = pass
print(f"Reason:  {strict_citation.reason}")
print(f"Passed:  {strict_citation.is_successful()}")  # True

# ==========================================
# 3. MISATTRIBUTED CITATION — fail (score 1)
# ==========================================
# Completion-year claim cites [1], but [1] is the height
# passage. Misattribution, even though the claim itself
# is true. Expect 1 (fail).

misattributed_case = LLMTestCase(
    input="When was the Eiffel Tower completed?",
    actual_output="The Eiffel Tower was completed in 1889 [1].",
    retrieval_context=[
        "The Eiffel Tower stands 330 metres tall in Paris.",
        "The Eiffel Tower was completed in 1889 for the World Fair.",
    ],
)

strict_citation.measure(misattributed_case)
print("--- one misattribution ---")
print(f"Score:   {strict_citation.score}")          # 1 = fail
print(f"Reason:  {strict_citation.reason}")
print(f"Passed:  {strict_citation.is_successful()}")  # False