from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRecallMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation (each assistant turn's `retrieval_context`)
# and the `expected_outcome`, and answers three questions:
#
#   Q1 (Noul, weight 2): every sentence in expected_outcome
#                        can be attributed to the
#                        retrieval_context of the assistant
#                        turns
#   Q2 (Noul, weight 1): the retrieval_context of the
#                        assistant turns contains the key
#                        facts needed
#   Q3 (Score, weight 1): how much of expected_outcome can
#                          be attributed to retrieval_context
#                          (none → all)
#
# Final score = weighted mean of the three answers.
#
# Under `system_one` Jev judges the conversation as a
# WHOLE — there's no per-window scoring like the LLM
# chain does.

system_one_turn_contextual_recall = TurnContextualRecallMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
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
system_one_turn_contextual_recall.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_turn_contextual_recall.score}")
print(f"Confidence: {system_one_turn_contextual_recall.confidence}")
print(f"Reason:     {system_one_turn_contextual_recall.reason}")
print(f"Passed:     {system_one_turn_contextual_recall.is_successful()}")