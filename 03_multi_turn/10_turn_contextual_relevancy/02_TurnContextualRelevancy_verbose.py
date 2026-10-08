from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRelevancyMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the statements extracted from each retrieval_context
#     node per window
#   - the verdict (relevant / not) for each statement
# This is the fastest way to debug "the retriever is
# returning garbage". Verbose mode surfaces which
# statements got flagged as irrelevant.

turn_contextual_relevancy = TurnContextualRelevancyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — one irrelevant statement
# ==========================================
# The retrieval_context has two statements:
#   1) "All customers are eligible for a 30 day full refund"
#       — relevant to the shoe-fit question
#   2) "Our company was founded in 1998"                  — irrelevant
# So relevancy should be 1/2 = 0.5.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost. Our company was founded in 1998."
            ],
        ),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
turn_contextual_relevancy.measure(convo_test_case)
print("---")
print(f"Score:   {turn_contextual_relevancy.score}")          # 0.5
print(f"Reason:  {turn_contextual_relevancy.reason}")
print(f"Passed:  {turn_contextual_relevancy.is_successful()}")  # True (boundary)