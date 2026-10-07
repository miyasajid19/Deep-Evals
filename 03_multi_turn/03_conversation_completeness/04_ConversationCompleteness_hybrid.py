from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import ConversationCompletenessMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Conversation Completeness: the LLM STILL
# extracts the user intentions (this is the only LLM step
# that remains). For each intention, Jev (a System One
# model) answers one yes/no question: "did the assistant
# satisfy this intention across the conversation?" with
# P(yes) >= 0.5 counted as satisfied. The equation and
# the LLM-written reason are unchanged.
#
# If a Jev call fails, the metric silently defers that
# decision to the LLM.
#
# Hybrid is the cheap/fast lane: the LLM does ONE
# extraction call (not per-intention), and Jev handles
# the per-intention satisfaction verdicts.

hybrid_conversation_completeness = ConversationCompletenessMetric(
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still extracts user intentions
    include_reason=True,
    eval_mode="hybrid",         # Jev judges per-intention satisfaction
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same support conversation as the basic run — every
# intention is satisfied.

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
# metric.confidence reports how confident the system_one
# model was on average across its calls — useful for
# debugging borderline cases.

hybrid_conversation_completeness.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_conversation_completeness.score}")
print(f"Confidence: {hybrid_conversation_completeness.confidence}")
print(f"Reason:     {hybrid_conversation_completeness.reason}")
print(f"Passed:     {hybrid_conversation_completeness.is_successful()}")