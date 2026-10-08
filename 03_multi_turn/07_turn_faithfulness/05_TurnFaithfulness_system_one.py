from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnFaithfulnessMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation — each turn carrying its `role`,
# `content`, and `retrieval_context` — and answers
# three questions:
#
#   Q1 (Noul, weight 2): every factual claim in an
#                        assistant turn is supported by
#                        the retrieval_context of that
#                        turn or an earlier assistant turn
#   Q2 (Noul, weight 1): no assistant turn contradicts
#                        its retrieval_context
#   Q3 (Score, weight 1): across turns, how much of what
#                          the assistant states is
#                          grounded (fabricated → fully
#                          grounded)
#
# Final score = weighted mean of the three answers.

system_one_turn_faithfulness = TurnFaithfulnessMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same RAG Q&A as the basic run.

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
# metric.confidence reflects how decisive Jev was. The
# reason lists each question's answer with its
# probability — useful when you want to know *why* Jev
# scored it that way, not just the final number.

system_one_turn_faithfulness.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_turn_faithfulness.score}")
print(f"Confidence: {system_one_turn_faithfulness.confidence}")
print(f"Reason:     {system_one_turn_faithfulness.reason}")
print(f"Passed:     {system_one_turn_faithfulness.is_successful()}")