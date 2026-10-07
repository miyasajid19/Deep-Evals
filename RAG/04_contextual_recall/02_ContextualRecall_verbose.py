from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the statements the LLM extracted from `expected_output`
#   - the verdict (attributable / not) for each statement
# This is the fastest way to debug a low score. If a fact
# the ideal answer relies on isn't in `retrieval_context`,
# you'll see which statement got flagged as not attributable.

contextual_recall = ContextualRecallMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — one statement is NOT attributable
# ==========================================
# Expected output has two sentences:
#   1) "You are eligible for a 30 day full refund at no extra cost." — supported
#   2) "You can initiate a return from your order page."           — NOT supported
# So contextual recall should be 1/2 ≈ 0.5.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output=(
        "You are eligible for a 30 day full refund at no extra cost. "
        "You can initiate a return from your order page."
    ),
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost."
        # The "initiate from order page" fact is missing — that's the leak.
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

contextual_recall.measure(test_case)
print("---")
print(f"Score:   {contextual_recall.score}")          # 0.5
print(f"Reason:  {contextual_recall.reason}")
print(f"Passed:  {contextual_recall.is_successful()}")  # False (under 0.7)