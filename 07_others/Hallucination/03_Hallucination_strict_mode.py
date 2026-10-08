from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if the actual_output aligned with every context
#   - 0 if any single context was contradicted
# It also overrides `threshold` to 1, so
# `is_successful()` matches the binary score exactly.
#
# Use it when any fabrication is unacceptable —
# compliance, medical, legal text generation.

strict_hallucination = HallucinationMetric(
    threshold=0.5,             # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. EVERY CONTEXT ALIGNED — score 1
# ==========================================
# Two contexts, both fully supported by the output.

perfect_case = LLMTestCase(
    input="What was the man wearing?",
    actual_output="A blond man in a brown shirt drinking water in public.",
    context=[
        "A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
        "The fountain is on the south side of the park, near the bench.",
    ],
)

strict_hallucination.measure(perfect_case)
print("--- all contexts aligned ---")
print(f"Score:   {strict_hallucination.score}")          # 1.0
print(f"Reason:  {strict_hallucination.reason}")
print(f"Passed:  {strict_hallucination.is_successful()}")  # True

# ==========================================
# 3. ONE CONTEXT CONTRADICTED — score 0
# ==========================================
# The output claims a red shirt, contradicting context
# [2]. Strict mode drops the score to 0 even though
# context [1] was fully aligned.

contradicted_case = LLMTestCase(
    input="What was the man wearing?",
    actual_output="A blond man in a red shirt drinking water in public.",
    context=[
        "A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
        "Witnesses confirmed his shirt was red, not brown.",
    ],
)

strict_hallucination.measure(contradicted_case)
print("--- one context contradicted ---")
print(f"Score:   {strict_hallucination.score}")          # 0.0
print(f"Reason:  {strict_hallucination.reason}")
print(f"Passed:  {strict_hallucination.is_successful()}")  # False