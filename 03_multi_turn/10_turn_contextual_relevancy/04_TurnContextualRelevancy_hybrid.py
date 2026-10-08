from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRelevancyMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Turn Contextual Relevancy: each node in an
# assistant turn's `retrieval_context` is split into
# sentences IN CODE (no LLM extraction step). Jev (a
# System One model) answers one yes/no question per
# statement: "is this relevant to the user's message?",
# with P(yes) >= 0.5 counted as relevant.
#
# The equation (per-window relevancy) and the LLM-written
# reason are unchanged. If a Jev call fails, the metric
# silently defers that decision to the LLM.

hybrid_turn_contextual_relevancy = TurnContextualRelevancyMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="hybrid",         # sentences split in code, Jev rates each
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Both statements on-topic.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost. Returns can be initiated from your order page.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost.",
                "Returns can be initiated from the customer's order page in under a minute.",
            ],
        ),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
hybrid_turn_contextual_relevancy.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_turn_contextual_relevancy.score}")
print(f"Confidence: {hybrid_turn_contextual_relevancy.confidence}")
print(f"Reason:     {hybrid_turn_contextual_relevancy.reason}")
print(f"Passed:     {hybrid_turn_contextual_relevancy.is_successful()}")