from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid skips the LLM extraction step entirely — the
# sliding windows are built in code, and Jev (a System
# One model) answers one yes/no question per window:
# "is the last assistant turn relevant to the conversation
# so far?" with P(yes) >= 0.5 counted as relevant. The
# equation and the LLM-written reason are unchanged.
#
# If a Jev call fails, the metric silently defers that
# decision to the LLM.
#
# Hybrid is the cheap/fast lane: no LLM extraction, just
# Jev verdicts. It's most useful on long conversations
# where the per-window LLM calls would dominate cost.

hybrid_turn_relevancy = TurnRelevancyMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="hybrid",         # Jev judges per-window, not the LLM
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same clean Q&A as the basic run — every assistant
# turn addresses the user's question. With
# `eval_mode="hybrid"`, the metric builds the windows
# locally and asks Jev per window.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(role="assistant", content="We offer a 30-day full refund at no extra cost."),
        Turn(role="user",      content="How do I start the return?"),
        Turn(role="assistant", content="You can initiate a return from your order page in under a minute."),
        Turn(role="user",      content="Will I get a shipping label?"),
        Turn(role="assistant", content="Yes — once you submit the return, we email you a prepaid shipping label."),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reports how confident the system_one
# model was on average across its calls — useful for
# debugging borderline cases.

hybrid_turn_relevancy.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_turn_relevancy.score}")
print(f"Confidence: {hybrid_turn_relevancy.confidence}")
print(f"Reason:     {hybrid_turn_relevancy.reason}")
print(f"Passed:     {hybrid_turn_relevancy.is_successful()}")