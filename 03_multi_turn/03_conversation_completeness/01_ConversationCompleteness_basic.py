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
        Turn(role="assistant", content="Your refund was processed on Monday."),
        Turn(role="user",      content="then what is the status tell everything at once?"),
        Turn(role="assistant", content="Its shipped."),
        Turn(role="user",      content="Tell everything at once. when will i get it?"),
        Turn(role="assistant", content="you will get it "),
    ]
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[convo_test_case], metrics=[conversation_completeness])