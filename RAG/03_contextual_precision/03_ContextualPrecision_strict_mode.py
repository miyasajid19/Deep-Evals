from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if WCP == 1 (every relevant node is ranked ahead
#     of every irrelevant node)
#   - 0 otherwise
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when ANY
# misplaced chunk is unacceptable (compliance, legal,
# tightly-scoped RAG bots).

strict_contextual_precision = ContextualPrecisionMetric(
    threshold=0.7,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. PERFECTLY RANKED — expect score 1
# ==========================================
# All relevant nodes (chunk 1 and chunk 2) are ranked
# ahead of the unrelated one (chunk 3). WCP = 1.0.

perfect_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Returns can be initiated from the customer's order page in under a minute.",
        "Our company was founded in 1998 and is privately owned.",
    ],
)

strict_contextual_precision.measure(perfect_case)
print("--- perfect ranking ---")
print(f"Score:   {strict_contextual_precision.score}")          # 1.0
print(f"Reason:  {strict_contextual_precision.reason}")
print(f"Passed:  {strict_contextual_precision.is_successful()}")  # True

# ==========================================
# 3. MIS-RANKED — expect score 0
# ==========================================
# The relevant node is BURIED below an unrelated one.
# WCP = 1/2 * 1 = 0.5, divided by 1 → 0.5. Strict mode
# rounds that to 0. Compare against the non-strict
# `02_verbose` run, where this same ranking would still
# pass under a 0.5 threshold but fail under 0.7.

misranked_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=[
        "Our company was founded in 1998 and is privately owned.",                   # noise ranked #1
        "All customers are eligible for a 30 day full refund at no extra cost.",    # relevant ranked #2
    ],
)

strict_contextual_precision.measure(misranked_case)
print("--- mis-ranked ---")
print(f"Score:   {strict_contextual_precision.score}")          # 0.0
print(f"Reason:  {strict_contextual_precision.reason}")
print(f"Passed:  {strict_contextual_precision.is_successful()}")  # False