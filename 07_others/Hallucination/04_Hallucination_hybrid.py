from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Hallucination: the LLM extraction step is
# preserved. The actual judgement — "does the
# actual_output agree with this context?" — is delegated
# to Jev, one yes/no question per context, with
# `P(yes) >= 0.5` counted as aligned.
#
# The equation is unchanged:
#   Score = N aligned / N total
# And the LLM-written reason is unchanged. If a Jev
# call fails, the metric silently defers that decision
# to the LLM.

hybrid_hallucination = HallucinationMetric(
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still extracts verdict context
    include_reason=True,
    eval_mode="hybrid",         # Jev classifies each context
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Two contexts, both fully aligned. Expect 2/2 = 1.0.

test_case = LLMTestCase(
    input="What was the man wearing?",
    actual_output="A blond man in a brown shirt drinking water in public.",
    context=[
        "A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
        "The fountain is on the south side of the park, near the bench.",
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# `metric.confidence` reports how decisive Jev was on
# average across its per-context calls.

hybrid_hallucination.measure(test_case)
print("---")
print(f"Score:      {hybrid_hallucination.score}")
print(f"Confidence: {hybrid_hallucination.confidence}")
print(f"Reason:     {hybrid_hallucination.reason}")
print(f"Passed:     {hybrid_hallucination.is_successful()}")