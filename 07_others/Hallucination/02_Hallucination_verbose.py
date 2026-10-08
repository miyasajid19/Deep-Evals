from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the per-context
# contradiction verdict (aligned vs contradicts) before
# the final score. This is the fastest way to debug
# "the output contradicted context #2" — verbose
# surfaces which context was flagged.

hallucination = HallucinationMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — two contexts, one contradicted
# ==========================================
# Context [1] is fully aligned with the output (a blond
# drinking water). Context [2] says the man was wearing
# a red shirt, but the output says brown shirt — that's
# a contradiction. Expect 1/2 = 0.5.

test_case = LLMTestCase(
    input="What was the blond doing and what was he wearing?",
    actual_output="A blond man in a brown shirt drinking water in public.",
    context=[
        "A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
        "Witnesses at the scene later described his shirt as bright red, not brown.",
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason.
# Verbose output prints the per-context verdicts above.

hallucination.measure(test_case)
print("---")
print(f"Score:   {hallucination.score}")          # 1/2 = 0.5
print(f"Reason:  {hallucination.reason}")
print(f"Passed:  {hallucination.is_successful()}")  # True (boundary)