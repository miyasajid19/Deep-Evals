from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the verdict (yes / no) for each node in order
#   - the weighted cumulative precision score itself
# This is the fastest way to debug a low score. If the
# LLM marks a useful node as "no", or doesn't penalize
# a noise node ranked first, you'll see the exact list
# it scored against.

contextual_precision = ContextualPrecisionMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — relevant first, noise later
# ==========================================
# The first node supports the expected answer; the last two
# are unrelated. Order matters: relevant on top, noise on
# the bottom is the *right* ordering for the WCP formula.
#
# WCP at n=3 with verdicts [yes, no, no]:
#   k=1: 1/1 * 1 = 1.0
#   k=2: 1/2 * 0 = 0.0
#   k=3: 1/3 * 0 = 0.0
#   sum  = 1.0  →  divided by # relevant (1)  →  1.0

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",   # relevant
        "Our company was founded in 1998 and is privately owned.",                    # unrelated
        "We also sell socks, hats, and jackets in our online store.",                 # unrelated
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

contextual_precision.measure(test_case)
print("---")
print(f"Score:   {contextual_precision.score}")          # 1.0 (perfect ordering)
print(f"Reason:  {contextual_precision.reason}")
print(f"Passed:  {contextual_precision.is_successful()}")  # True