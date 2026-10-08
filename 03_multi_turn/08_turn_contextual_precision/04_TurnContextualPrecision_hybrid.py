from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualPrecisionMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Turn Contextual Precision works on the LLM
# judgement step (just like the single-turn hybrid mode):
# the LLM doesn't do per-node relevance calls at all. Jev
# (a System One model) answers one yes/no question per
# node: "is this node useful for arriving at the
# expected outcome?", with P(yes) >= 0.5 counted as
# useful.
#
# The equation (per-window WCP) and the LLM-written
# reason are unchanged. If a Jev call fails, the metric
# silently defers that decision to the LLM.

hybrid_turn_contextual_precision = TurnContextualPrecisionMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="hybrid",         # Jev answers per-node relevance
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same setup as the basic run — single useful node, two
# unrelated. With `eval_mode="hybrid"`, Jev judges each
# node's relevance.

convo_test_case = ConversationalTestCase(
    expected_outcome=(
        "The chatbot must explain store policies clearly, including "
        "refund eligibility, return process, and shipping timelines."
    ),
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost.",
                "Our company was founded in 1998 and is privately owned.",
            ],
        ),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reports how confident the system_one
# model was on average across its per-node calls.

hybrid_turn_contextual_precision.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_turn_contextual_precision.score}")
print(f"Confidence: {hybrid_turn_contextual_precision.confidence}")
print(f"Reason:     {hybrid_turn_contextual_precision.reason}")
print(f"Passed:     {hybrid_turn_contextual_precision.is_successful()}")