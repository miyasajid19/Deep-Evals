from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives `expected_output`
# and `retrieval_context` and answers three questions:
#
#   Q1 (Noul, weight 2): every sentence in expected_output
#                        can be attributed to the context
#   Q2 (Noul, weight 1): context contains the key facts
#   Q3 (Score, weight 1): how much of expected_output is
#                          attributable (none → all)
#
# Final score = weighted mean of the three answers.
#
# system_one mode is the right pick when you want the
# cheapest possible judge (no extraction LLM, no per-
# sentence calls) and don't need to compare against an
# LLM-mode score from the same metric.

system_one_contextual_recall = ContextualRecallMetric(
    threshold=0.7,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same on-target setup as the basic run.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost."
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reflects how decisive Jev was. The
# reason lists each question's answer with its
# probability — useful when you want to know *why* Jev
# scored it that way, not just the final number.

system_one_contextual_recall.measure(test_case)
print("---")
print(f"Score:      {system_one_contextual_recall.score}")
print(f"Confidence: {system_one_contextual_recall.confidence}")
print(f"Reason:     {system_one_contextual_recall.reason}")
print(f"Passed:     {system_one_contextual_recall.is_successful()}")