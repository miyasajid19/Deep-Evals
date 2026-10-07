from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import ConversationCompletenessMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# ConversationCompletenessMetric measures whether the
# assistant satisfies EVERY user intention raised in a
# conversation. It's a useful proxy for user
# satisfaction in chatbot flows where one user can voice
# several requests in a row.
#
# Algorithm (default `llm` eval mode):
#   1) extract high-level user intentions from user turns
#   2) for each intention, judge whether the assistant
#      satisfied it across the conversation
#   3) score = (# satisfied intentions) /
#              (# total intentions)
#
# The denominator is user intentions, not assistant
# turns — a single missed intention costs you a full
# numerator slot. The metric is `referenceless`: no
# `expected_output` needed.

conversation_completeness = ConversationCompletenessMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# A support conversation where the user voices three
# intentions (refund status, return shipping, account
# email change) and the assistant addresses every one.
# Expect score 1.0.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I want to check the status of my refund from last week."),
        Turn(role="assistant", content="Your refund was processed on Monday and should land within 3-5 business days."),
        Turn(role="user",      content="Great. Also, how do I set up a return for a different order?"),
        Turn(role="assistant", content="You can start a return from your order history page — pick the order, hit 'Return Item', and we'll email you a prepaid label."),
        Turn(role="user",      content="Last thing — I need to change the email on my account from ali@old.com to ali@new.com."),
        Turn(role="assistant", content="Done — your account email is now ali@new.com. I kept ali@old.com as a forwarding alias for 30 days just in case."),
    ]
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[convo_test_case], metrics=[conversation_completeness])