from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if every claim in `actual_output` is faithful
#   - 0 if any single claim contradicts `retrieval_context`
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when a single
# unsupported claim is unacceptable (medical, legal,
# compliance RAG bots).

strict_faithfulness = FaithfulnessMetric(
    threshold=0.7,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. FULLY FAITHFUL — expect score 1
# ==========================================
# Every claim is supported by the retrieved context.
# The metric should mark the answer as fully faithful.

perfect_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "We offer a 30-day full refund at no extra cost. "
        "You can initiate a return from your order page."
    ),
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Returns can be initiated from the customer's order page.",
    ],
)

strict_faithfulness.measure(perfect_case)
print("--- perfect case ---")
print(f"Score:   {strict_faithfulness.score}")          # 1.0
print(f"Reason:  {strict_faithfulness.reason}")
print(f"Passed:  {strict_faithfulness.is_successful()}")  # True

# ==========================================
# 3. ONE CONTRADICTION — expect score 0
# ==========================================
# A single contradicting claim flips the whole metric to 0.
# Compare against the non-strict `02_verbose` run, where
# the same answer would still score 2/3 (≈ 0.667).

contradicting_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "We offer a 30-day full refund at no extra cost. "
        "A $10 restocking fee applies to all returns."
    ),
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
    ],
)

strict_faithfulness.measure(contradicting_case)
print("--- contradicting case ---")
print(f"Score:   {strict_faithfulness.score}")          # 0.0
print(f"Reason:  {strict_faithfulness.reason}")
print(f"Passed:  {strict_faithfulness.is_successful()}")  # False