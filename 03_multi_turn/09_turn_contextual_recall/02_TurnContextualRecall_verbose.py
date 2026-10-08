from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRecallMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the statements extracted from `expected_outcome`
#   - the verdict (attributable / not) for each statement
#     per window
# This is the fastest way to debug "the retriever missed
# something the bot needed". Verbose mode surfaces which
# statement got flagged as not attributable.

turn_contextual_recall = TurnContextualRecallMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — one statement is NOT attributable
# ==========================================
# Expected outcome has two statements:
#   1) "refund eligibility at no extra cost"           — supported
#   2) "shipping timelines"                              — NOT supported
# So recall should be 1/2 = 0.5.

convo_test_case = ConversationalTestCase(
    expected_outcome=(
        "The chatbot must explain store policies clearly, including "
        "refund eligibility and shipping timelines."
    ),
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                # Only refund eligibility is here — shipping is the leak.
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
        ),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
turn_contextual_recall.measure(convo_test_case)
print("---")
print(f"Score:   {turn_contextual_recall.score}")          # 0.5
print(f"Reason:  {turn_contextual_recall.reason}")
print(f"Passed:  {turn_contextual_recall.is_successful()}")  # True (boundary)