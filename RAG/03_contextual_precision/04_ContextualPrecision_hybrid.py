from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid mode here doesn't have an LLM extraction step
# — Contextual Precision only does one LLM call (the
# per-node verdict). Under `eval_mode="hybrid"`, that
# per-node verdict is answered by Jev, a System One
# model: one yes/no question per node, with P(yes) >= 0.5
# counted as relevant. The WCP equation and the reason
# are unchanged.
#
# Hybrid is the cheap/fast lane: a single Jev call per
# node replaces the LLM call. If a Jev call fails, the
# metric silently defers that decision to the LLM.

hybrid_contextual_precision = ContextualPrecisionMetric(
    threshold=0.7,
    include_reason=True,
    eval_mode="hybrid",         # Jev judges relevance, not the LLM
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same Q&A as the basic run — single relevant node.
# Order doesn't matter when there's only one node, but
# the metric still asks the per-node Jev question.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Our company was founded in 1998 and is privately owned.",
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reports how confident the system_one
# model was on average across its calls — useful for
# debugging borderline cases.

hybrid_contextual_precision.measure(test_case)
print("---")
print(f"Score:      {hybrid_contextual_precision.score}")
print(f"Confidence: {hybrid_contextual_precision.confidence}")
print(f"Reason:     {hybrid_contextual_precision.reason}")
print(f"Passed:     {hybrid_contextual_precision.is_successful()}")