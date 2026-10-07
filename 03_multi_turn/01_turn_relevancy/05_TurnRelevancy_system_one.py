from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation and answers three questions:
#
#   Q1 (Noul, weight 2): every assistant turn is relevant
#   Q2 (Noul, weight 1): no assistant turn ignores its
#                        user turn or changes the subject
#   Q3 (Score, weight 1): across turns, how relevant
#                          are the assistant replies
#                          (mostly irrelevant → all relevant)
#
# Final score = weighted mean of the three answers.
#
# Under `system_one` Jev judges the conversation as a
# WHOLE — there's no per-window scoring like the LLM
# chain does. Pick this mode when you want the cheapest
# possible judge and don't need to compare against an
# LLM-mode score from the same metric.

system_one_turn_relevancy = TurnRelevancyMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same on-target setup as the basic run.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(role="assistant", content="We offer a 30-day full refund at no extra cost."),
        Turn(role="user",      content="How do I start the return?"),
        Turn(role="assistant", content="You can initiate a return from your order page in under a minute."),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reflects how decisive Jev was. The
# reason lists each question's answer with its
# probability — useful when you want to know *why* Jev
# scored it that way, not just the final number.

system_one_turn_relevancy.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_turn_relevancy.score}")
print(f"Confidence: {system_one_turn_relevancy.confidence}")
print(f"Reason:     {system_one_turn_relevancy.reason}")
print(f"Passed:     {system_one_turn_relevancy.is_successful()}")