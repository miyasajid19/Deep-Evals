from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the statements extracted from the retrieval_context
#   - the verdict (relevant / not) for each statement
# This is the fastest way to debug "the retriever is
# returning garbage". Verbose mode surfaces which
# statement got flagged.

contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — one irrelevant statement
# ==========================================
# Two statements:
#   1) "refund eligibility at no extra cost" — relevant
#   2) "company founded in 1998"              — irrelevant
# Expect 1/2 = 0.5.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost. Our company was founded in 1998."
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
contextual_relevancy.measure(test_case)
print("---")
print(f"Score:   {contextual_relevancy.score}")          # 0.5
print(f"Reason:  {contextual_relevancy.reason}")
print(f"Passed:  {contextual_relevancy.is_successful()}")  # False (under 0.7)