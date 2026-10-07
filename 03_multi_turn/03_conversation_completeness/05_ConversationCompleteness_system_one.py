from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import ConversationCompletenessMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation and answers three questions:
#
#   Q1 (Noul, weight 2): every user intention is
#                        satisfied by the end
#   Q2 (Noul, weight 1): no user request is ignored,
#                        refused without cause, or only
#                        partly addressed
#   Q3 (Score, weight 1): across the conversation, how
#                        many intentions the assistant
#                        satisfied (none → all)
#
# Final score = weighted mean of the three answers.
#
# system_one mode is the right pick when you want the
# cheapest possible judge and don't need to compare
# against an LLM-mode score from the same metric.

system_one_conversation_completeness = ConversationCompletenessMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same support conversation as the basic run.

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
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reflects how decisive Jev was. The
# reason lists each question's answer with its
# probability — useful when you want to know *why* Jev
# scored it that way, not just the final number.

system_one_conversation_completeness.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_conversation_completeness.score}")
print(f"Confidence: {system_one_conversation_completeness.confidence}")
print(f"Reason:     {system_one_conversation_completeness.reason}")
print(f"Passed:     {system_one_conversation_completeness.is_successful()}")