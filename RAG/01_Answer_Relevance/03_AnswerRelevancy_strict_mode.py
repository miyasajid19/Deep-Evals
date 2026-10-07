from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if the LLM judges the answer perfectly relevant
#   - 0 otherwise (any off-topic statement flips it)
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when a single
# irrelevant sentence is unacceptable (compliance, legal,
# tightly-scoped RAG bots).

strict_relevancy = AnswerRelevancyMetric(
    threshold=0.7,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. PERFECTLY RELEVANT — expect score 1
# ==========================================
# Every statement is on-topic, so the LLM should mark the
# answer as fully relevant.

perfect_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "We offer a 30-day full refund at no extra cost. "
        "You can initiate a return from your order page."
    ),
)

strict_relevancy.measure(perfect_case)
print("--- perfect case ---")
print(f"Score:   {strict_relevancy.score}")          # 1.0
print(f"Reason:  {strict_relevancy.reason}")
print(f"Passed:  {strict_relevancy.is_successful()}")  # True

# ==========================================
# 3. ONE OFF-TOPIC SENTENCE — expect score 0
# ==========================================
# Strict mode punishes even a single filler sentence.
# Compare against the non-strict `02_verbose` run, where
# this same answer would still pass 0.7.

# Reset internal state by creating a new test case on the same metric.
mixed_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "We offer a 30-day full refund at no extra cost. "
        "We also sell socks, hats, and jackets."
    ),
)

strict_relevancy.measure(mixed_case)
print("--- mixed case ---")
print(f"Score:   {strict_relevancy.score}")          # 0.0
print(f"Reason:  {strict_relevancy.reason}")
print(f"Passed:  {strict_relevancy.is_successful()}")  # False