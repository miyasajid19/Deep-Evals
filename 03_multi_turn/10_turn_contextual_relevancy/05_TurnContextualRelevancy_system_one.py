from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRelevancyMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation (each assistant turn's `retrieval_context`)
# and answers three questions:
#
#   Q1 (Noul, weight 2): every statement in the
#                        retrieval_context of each assistant
#                        turn is relevant to the user turns
#                        before it
#   Q2 (Noul, weight 1): every node in the
#                        retrieval_context of each assistant
#                        turn contains at least one relevant
#                        statement
#   Q3 (Score, weight 1): across turns, how much of the
#                          retrieval_context is relevant
#                          (none → all)
#
# Final score = weighted mean of the three answers.
#
# Under `system_one` Jev judges the conversation as a
# WHOLE — there's no per-window scoring like the LLM
# chain does.

system_one_turn_contextual_relevancy = TurnContextualRelevancyMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
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
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
system_one_turn_contextual_relevancy.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_turn_contextual_relevancy.score}")
print(f"Confidence: {system_one_turn_contextual_relevancy.confidence}")
print(f"Reason:     {system_one_turn_contextual_relevancy.reason}")
print(f"Passed:     {system_one_turn_contextual_relevancy.is_successful()}")