from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if EVERY statement in `expected_output` can be
#     attributed to `retrieval_context`
#   - 0 otherwise (any unattributable statement flips it)
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when missing
# even one supporting chunk is unacceptable (compliance,
# legal, tightly-scoped RAG bots).

strict_contextual_recall = ContextualRecallMetric(
    threshold=0.7,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. FULLY ATTRIBUTABLE — expect score 1
# ==========================================
# Both expected statements trace back to nodes in the
# retrieved context.

perfect_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output=(
        "You are eligible for a 30 day full refund at no extra cost. "
        "You can initiate a return from your order page."
    ),
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Returns can be initiated from the customer's order page in under a minute.",
    ],
)

strict_contextual_recall.measure(perfect_case)
print("--- fully attributable ---")
print(f"Score:   {strict_contextual_recall.score}")          # 1.0
print(f"Reason:  {strict_contextual_recall.reason}")
print(f"Passed:  {strict_contextual_recall.is_successful()}")  # True

# ==========================================
# 3. PARTIAL COVERAGE — expect score 0
# ==========================================
# The "initiate from order page" sentence is missing from
# the retrieval_context. Strict mode drops the score to 0.
# Compare against the non-strict `02_verbose` run, where
# the same answer would still score 0.5.

partial_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output=(
        "You are eligible for a 30 day full refund at no extra cost. "
        "You can initiate a return from your order page."
    ),
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
    ],
)

strict_contextual_recall.measure(partial_case)
print("--- partial coverage ---")
print(f"Score:   {strict_contextual_recall.score}")          # 0.0
print(f"Reason:  {strict_contextual_recall.reason}")
print(f"Passed:  {strict_contextual_recall.is_successful()}")  # False