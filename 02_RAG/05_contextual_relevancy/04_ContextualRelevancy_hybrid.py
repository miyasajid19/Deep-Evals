from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Contextual Relevancy: `retrieval_context` is
# split into sentences IN CODE (no LLM extraction step at
# all). Jev (a System One model) answers one yes/no
# question per sentence: "is this statement relevant to the
# input?", with P(yes) >= 0.5 counted as relevant.
#
# The equation (# relevant / # total) and the LLM-written
# reason are unchanged. If a Jev call fails, the metric
# silently defers that decision to the LLM.

hybrid_contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    include_reason=True,
    eval_mode="hybrid",         # sentences split in code, Jev rates each
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Both retrieval_context statements relevant.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost. You can initiate a return from your order page.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Returns can be initiated from the customer's order page in under a minute.",
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
hybrid_contextual_relevancy.measure(test_case)
print("---")
print(f"Score:      {hybrid_contextual_relevancy.score}")
print(f"Confidence: {hybrid_contextual_relevancy.confidence}")
print(f"Reason:     {hybrid_contextual_relevancy.reason}")
print(f"Passed:     {hybrid_contextual_relevancy.is_successful()}")