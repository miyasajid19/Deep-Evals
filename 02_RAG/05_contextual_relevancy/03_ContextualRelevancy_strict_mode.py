from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if every retrieval_context statement is relevant
#   - 0 if any single statement is irrelevant
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# noisy chunk breaks the experience (compliance, legal,
# tightly-scoped RAG bots).

strict_contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. EVERY STATEMENT RELEVANT — expect score 1
# ==========================================
perfect_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost. You can initiate a return from your order page.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Returns can be initiated from the customer's order page in under a minute.",
    ],
)

strict_contextual_relevancy.measure(perfect_case)
print("--- every statement relevant ---")
print(f"Score:   {strict_contextual_relevancy.score}")          # 1.0
print(f"Reason:  {strict_contextual_relevancy.reason}")
print(f"Passed:  {strict_contextual_relevancy.is_successful()}")  # True

# ==========================================
# 3. ONE IRRELEVANT STATEMENT — expect score 0
# ==========================================
# The retrieval_context contains a noise statement.
# Strict mode drops the score to 0.

noisy_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost. Our company was founded in 1998."
    ],
)

strict_contextual_relevancy.measure(noisy_case)
print("--- one irrelevant statement ---")
print(f"Score:   {strict_contextual_relevancy.score}")          # 0.0
print(f"Reason:  {strict_contextual_relevancy.reason}")
print(f"Passed:  {strict_contextual_relevancy.is_successful()}")  # False