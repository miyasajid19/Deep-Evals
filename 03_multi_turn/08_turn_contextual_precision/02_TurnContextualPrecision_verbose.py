from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualPrecisionMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the verdict (useful / not) for each retrieval_context
#     node in each window
#   - the weighted cumulative precision score per window
# This is the fastest way to debug a low score. If a
# useful node gets marked "not useful" or a noise node
# ranks first, verbose mode will surface it.

turn_contextual_precision = TurnContextualPrecisionMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — relevant node ranked first
# ==========================================
# Three retrieval_context nodes. The first is useful
# (refund policy); the second and third are unrelated.
# Verdicts: [yes, no, no].
#
# WCP at n=3:
#   k=1: 1/1 * 1 = 1.0
#   k=2: 1/2 * 0 = 0.0
#   k=3: 1/3 * 0 = 0.0
#   sum  = 1.0  →  divided by # useful (1)  →  1.0

convo_test_case = ConversationalTestCase(
    expected_outcome=(
        "The chatbot must explain store policies clearly, including "
        "refund eligibility, return process, and shipping timelines."
    ),
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost.",   # useful
                "Our company was founded in 1998 and is privately owned.",                    # unrelated
                "We also sell socks, hats, and jackets in our online store.",                 # unrelated
            ],
        ),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

turn_contextual_precision.measure(convo_test_case)
print("---")
print(f"Score:   {turn_contextual_precision.score}")          # 1.0
print(f"Reason:  {turn_contextual_precision.reason}")
print(f"Passed:  {turn_contextual_precision.is_successful()}")  # True