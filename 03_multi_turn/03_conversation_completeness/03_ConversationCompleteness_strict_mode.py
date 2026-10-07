from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import ConversationCompletenessMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if EVERY user intention is satisfied
#   - 0 if any single user intention is missed
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# dropped request is unacceptable (compliance, legal,
# tight support bots where "anything else?" isn't a
# valid close).

strict_conversation_completeness = ConversationCompletenessMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. EVERY INTENTION SATISFIED — expect score 1
# ==========================================
# Three user intentions, three satisfied replies.

perfect_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I want to check the status of my refund from last week."),
        Turn(role="assistant", content="Your refund was processed on Monday and should land within 3-5 business days."),
        Turn(role="user",      content="Great. Also, how do I set up a return for a different order?"),
        Turn(role="assistant", content="You can start a return from your order history page — pick the order, hit 'Return Item', and we'll email you a prepaid label."),
        Turn(role="user",      content="Last thing — I need to change the email on my account from ali@old.com to ali@new.com."),
        Turn(role="assistant", content="Done — your account email is now ali@new.com. I kept ali@old.com as a forwarding alias for 30 days just in case."),
    ]
)

strict_conversation_completeness.measure(perfect_case)
print("--- every intention satisfied ---")
print(f"Score:   {strict_conversation_completeness.score}")          # 1.0
print(f"Reason:  {strict_conversation_completeness.reason}")
print(f"Passed:  {strict_conversation_completeness.is_successful()}")  # True

# ==========================================
# 3. ONE INTENTION MISSED — expect score 0
# ==========================================
# The assistant skips the email-change request and ends
# with "anything else?". Strict mode drops the score to
# 0. Compare against the non-strict `02_verbose` run,
# where the same conversation scores 2/3 ≈ 0.667.

missed_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I want to check the status of my refund from last week."),
        Turn(role="assistant", content="Your refund was processed on Monday and should land within 3-5 business days."),
        Turn(role="user",      content="Great. Also, how do I set up a return for a different order?"),
        Turn(role="assistant", content="You can start a return from your order history page — pick the order, hit 'Return Item', and we'll email you a prepaid label."),
        Turn(role="user",      content="Last thing — I need to change the email on my account from ali@old.com to ali@new.com."),
        Turn(role="assistant", content="Glad I could help with the return! Is there anything else I can do for you today?"),  # ignored
    ]
)

strict_conversation_completeness.measure(missed_case)
print("--- one intention missed ---")
print(f"Score:   {strict_conversation_completeness.score}")          # 0.0
print(f"Reason:  {strict_conversation_completeness.reason}")
print(f"Passed:  {strict_conversation_completeness.is_successful()}")  # False