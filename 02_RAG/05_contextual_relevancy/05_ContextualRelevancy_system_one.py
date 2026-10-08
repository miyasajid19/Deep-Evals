from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the raw
# `retrieval_context` (with `input` and `actual_output`
# for context) and answers three questions:
#
#   Q1 (Noul, weight 2): every sentence in retrieval_context
#                        is relevant to addressing input
#   Q2 (Noul, weight 1): every retrieval_context node
#                        contains at least one relevant
#                        statement
#   Q3 (Score, weight 1): how much of retrieval_context is
#                          relevant (none → all)
#
# Final score = weighted mean of the three answers.

system_one_contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Single fully relevant retrieval node.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost."
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
system_one_contextual_relevancy.measure(test_case)
print("---")
print(f"Score:      {system_one_contextual_relevancy.score}")
print(f"Confidence: {system_one_contextual_relevancy.confidence}")
print(f"Reason:     {system_one_contextual_relevancy.reason}")
print(f"Passed:     {system_one_contextual_relevancy.is_successful()}")