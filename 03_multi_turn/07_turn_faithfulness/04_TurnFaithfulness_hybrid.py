from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnFaithfulnessMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Turn Faithfulness: the LLM still extracts
# truths from the retrieval_context and claims from the
# assistant reply. What hybrid *does* change is the
# per-claim classification:
#
#   For each claim, Jev (a System One model) answers
#   one yes/no question: "does this claim contradict the
#   truths from the window's retrieval_context?" with
#   P(yes) > 0.65 counted as truthful, < 0.35 as
#   contradictory, anything in between as borderline.
#
# The extraction, the per-window equation, and the reason
# are unchanged. If a Jev call fails, the metric
# silently defers that decision to the LLM.

hybrid_turn_faithfulness = TurnFaithfulnessMetric(
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still extracts truths + claims
    include_reason=True,
    eval_mode="hybrid",         # Jev classifies each claim
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same RAG Q&A as the basic run — every claim grounded.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
        ),
        Turn(role="user",      content="How do I start the return?"),
        Turn(
            role="assistant",
            content="You can initiate a return from your order page in under a minute.",
            retrieval_context=[
                "Returns can be initiated from the customer's order page in under a minute."
            ],
        ),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reports how confident the system_one
# model was on average across its per-claim calls.

hybrid_turn_faithfulness.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_turn_faithfulness.score}")
print(f"Confidence: {hybrid_turn_faithfulness.confidence}")
print(f"Reason:     {hybrid_turn_faithfulness.reason}")
print(f"Passed:     {hybrid_turn_faithfulness.is_successful()}")