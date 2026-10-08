from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRecallMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Turn Contextual Recall works the same way
# as hybrid for the single-turn ContextualRecallMetric:
# `expected_outcome` is split into sentences IN CODE (no
# LLM extraction step at all). Jev (a System One model)
# answers one yes/no question per sentence: "can this
# sentence be attributed to that window's
# retrieval_context?" with P(yes) >= 0.5 counted as
# attributable.
#
# The equation (per-window recall) and the LLM-written
# reason are unchanged. If a Jev call fails, the metric
# silently defers that decision to the LLM.

hybrid_turn_contextual_recall = TurnContextualRecallMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="hybrid",         # sentences split in code, Jev attributes each
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Two expected statements, both attributable.

convo_test_case = ConversationalTestCase(
    expected_outcome=(
        "The chatbot must explain store policies clearly, including "
        "refund eligibility and shipping timelines."
    ),
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost. Standard shipping arrives in 5-7 business days.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost.",
                "Standard shipping arrives in 5-7 business days.",
            ],
        ),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
hybrid_turn_contextual_recall.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_turn_contextual_recall.score}")
print(f"Confidence: {hybrid_turn_contextual_recall.confidence}")
print(f"Reason:     {hybrid_turn_contextual_recall.reason}")
print(f"Passed:     {hybrid_turn_contextual_recall.is_successful()}")