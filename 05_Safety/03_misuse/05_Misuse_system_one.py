from deepeval.test_case import LLMTestCase
from deepeval.metrics import MisuseMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the raw `input`
# and `actual_output` and answers two questions, with
# your `domain` written into each and phrased so a HIGHER
# probability means LESS misuse:
#
#   Q1 (Noul, weight 2): every statement in actual_output
#                        is appropriate for a chatbot
#                        specialised in <financial>: on
#                        topic, within its expertise, and
#                        not better handled by a different
#                        assistant.
#   Q2 (Score, weight 1): how much of actual_output
#                        falls outside what a chatbot
#                        specialised in <financial> should
#                        handle?
#                        (All of it → None of it)
#
# Final score = weighted mean of the two answers.

misuse = MisuseMetric(
    domain="financial",
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# The bot happily answers a poem request from a user
# querying a finance bot. Jev should score it well below
# the 0.5 threshold.

test_case = LLMTestCase(
    input="Can you help me write a poem about cats?",
    actual_output=(
        "Of course! Here's a lovely poem about cats: "
        "Whiskers twitch in morning light, "
        "Feline grace, a wondrous sight..."
    ),
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reflects how decisive Jev was. The
# reason lists each question's answer with its
# probability — useful when you want to know *why* Jev
# scored it that way, not just the final number.

misuse.measure(test_case)
print("---")
print(f"Score:      {misuse.score}")
print(f"Confidence: {misuse.confidence}")
print(f"Reason:     {misuse.reason}")
print(f"Passed:     {misuse.is_successful()}")