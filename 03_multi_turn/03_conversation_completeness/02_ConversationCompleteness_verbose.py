from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import ConversationCompletenessMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the user intentions the LLM extracted
#   - the verdict (satisfied / not) for each intention
# This is the fastest way to debug "the bot said something
# polite but didn't actually do the thing". Verbose mode
# surfaces which intention got ignored and which reply
# failed to satisfy it.

conversation_completeness = ConversationCompletenessMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — third intention is ignored
# ==========================================
# Three user intentions:
#   1) check refund status        — satisfied
#   2) set up a return             — satisfied
#   3) change account email        — IGNORED (the
#      assistant says "anything else?" instead)
#
# Expect score 2/3 ≈ 0.667.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I want to check the status of my refund from last week."),
        Turn(role="assistant", content="Your refund was processed on Monday and should land within 3-5 business days."),
        Turn(role="user",      content="Great. Also, how do I set up a return for a different order?"),
        Turn(role="assistant", content="You can start a return from your order history page — pick the order, hit 'Return Item', and we'll email you a prepaid label."),
        Turn(role="user",      content="Last thing — I need to change the email on my account from ali@old.com to ali@new.com."),
        Turn(role="assistant", content="Glad I could help with the return! Is there anything else I can do for you today?"),  # ignored the email change
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

conversation_completeness.measure(convo_test_case)
print("---")
print(f"Score:   {conversation_completeness.score}")          # 2/3 ≈ 0.667
print(f"Reason:  {conversation_completeness.reason}")
print(f"Passed:  {conversation_completeness.is_successful()}")  # True (under 0.5 threshold)