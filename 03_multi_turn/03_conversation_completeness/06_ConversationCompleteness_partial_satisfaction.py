from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import ConversationCompletenessMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three support conversations
# of varying completeness. The user voices three
# intentions each time; the difference is whether the
# assistant satisfies all of them, drops one, or only
# partially handles the main one.
#
# The metric's denominator is user intentions, so each
# dropped request costs you one full numerator slot.

conversation_completeness = ConversationCompletenessMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED INTENTIONS
# ==========================================
# Every case below has the user voicing three intentions:
#   1) refund status
#   2) return setup
#   3) account email change

# ==========================================
# 3. CASE A — full coverage
# ==========================================
# Every intention is satisfied. Expect score 1.0.

full_case = ConversationalTestCase(
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
# 4. CASE B — one intention missed
# ==========================================
# The assistant handles the first two, then ends with
# "anything else?" without addressing the email change.
# Expect 2/3 ≈ 0.667.

one_missed_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I want to check the status of my refund from last week."),
        Turn(role="assistant", content="Your refund was processed on Monday and should land within 3-5 business days."),
        Turn(role="user",      content="Great. Also, how do I set up a return for a different order?"),
        Turn(role="assistant", content="You can start a return from your order history page — pick the order, hit 'Return Item', and we'll email you a prepaid label."),
        Turn(role="user",      content="Last thing — I need to change the email on my account from ali@old.com to ali@new.com."),
        Turn(role="assistant", content="Glad I could help with the return! Is there anything else I can do for you today?"),
    ]
)

# ==========================================
# 5. CASE C — partial satisfaction on main intent
# ==========================================
# The assistant only partially addresses the refund
# question ("I'll need to check that" but never does),
# fully handles the return, and ignores the email
# change. Expect roughly 1/3 ≈ 0.333.

partial_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I want to check the status of my refund from last week."),
        Turn(role="assistant", content="I'll need to check on that — let me get back to you within 24 hours."),  # not satisfied
        Turn(role="user",      content="Great. Also, how do I set up a return for a different order?"),
        Turn(role="assistant", content="You can start a return from your order history page — pick the order, hit 'Return Item', and we'll email you a prepaid label."),
        Turn(role="user",      content="Last thing — I need to change the email on my account from ali@old.com to ali@new.com."),
        Turn(role="assistant", content="Glad I could help with the return setup! Have a good one."),                  # ignored
    ]
)

# ==========================================
# 6. RUN THE METRIC ON ALL THREE CASES
# ==========================================
# Watch the score drop as the bot misses more requests.

def run(label, case):
    conversation_completeness.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {conversation_completeness.score:.3f}")
    print(f"  Passed: {conversation_completeness.is_successful()}")
    print(f"  Reason: {conversation_completeness.reason}")

run("A — full coverage",                  full_case)
run("B — one intention missed",           one_missed_case)
run("C — partial satisfaction",           partial_case)