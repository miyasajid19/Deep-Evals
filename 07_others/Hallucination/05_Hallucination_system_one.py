from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# Under `system_one` mode, Jev judges the whole metric
# in one request. No LLM is called at all. Jev is sent
# the raw `actual_output` and the joined `context` and
# asked three weighted questions:
#
#   Q1 (Noul, weight 2): actual_output agrees with
#                        every document in context.
#   Q2 (Noul, weight 1): no statement in actual_output
#                        contradicts a document in
#                        context.
#   Q3 (Score, weight 1): how much of actual_output is
#                        consistent with the documents
#                        in context? (Contradicts them
#                        → Fully consistent)
#
# Each answer becomes a value in [0, 1] and the score is
# their weighted mean. `metric.confidence` reports how
# decisive Jev was on average.
#
# NOTE: a system_one score comes from Jev's answers, not
# from the LLM-chain equation. Don't compare across
# eval modes.

system_one_hallucination = HallucinationMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Two contexts, both fully aligned with the output.
# Jev should answer every question positively.

test_case = LLMTestCase(
    input="What was the man wearing?",
    actual_output="A blond man in a brown shirt drinking water in public.",
    context=[
        "A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
        "The fountain is on the south side of the park, near the bench.",
    ],
)

# ==========================================
# 3. RUN STANDALONE
# =========================================_

system_one_hallucination.measure(test_case)
print("---")
print(f"Score:      {system_one_hallucination.score}")
print(f"Confidence: {system_one_hallucination.confidence}")
print(f"Reason:     {system_one_hallucination.reason}")
print(f"Passed:     {system_one_hallucination.is_successful()}")