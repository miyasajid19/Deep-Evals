from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the list of claims the LLM extracted from `actual_output`
#   - the verdict (faithful / contradicts) for each claim
# This is the fastest way to debug a low score. If the LLM
# missed a contradiction or split a single claim into two,
# you'll see the exact list it scored against.

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — one claim contradicts context
# ==========================================
# The first two sentences are supported by the retrieved
# context. The third one contradicts it (no restocking fee
# is mentioned anywhere). Verbose mode will show which
# claim was marked as a contradiction.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "We offer a 30-day full refund at no extra cost. "
        "You can initiate a return from your order page. "
        "A $10 restocking fee applies to all returns."
    ),
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Returns can be initiated from the customer's order page in under a minute.",
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

faithfulness.measure(test_case)
print("---")
print(f"Score:   {faithfulness.score}")          # 2/3 ≈ 0.667
print(f"Reason:  {faithfulness.reason}")
print(f"Passed:  {faithfulness.is_successful()}")  # False (under 0.7)