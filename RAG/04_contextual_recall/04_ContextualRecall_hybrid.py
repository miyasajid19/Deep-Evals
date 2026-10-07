from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Contextual Recall works a bit differently
# from the other hybrid modes: the LLM extraction step
# is skipped entirely — `expected_output` is split into
# sentences in code (no LLM call). Jev, a System One
# model, then answers one yes/no question per sentence:
# "can this sentence be attributed to retrieval_context?"
# with P(yes) >= 0.5 counted as attributable.
#
# The equation and the LLM-written reason are unchanged.
# If a Jev call fails, the metric silently defers that
# decision to the LLM.
#
# Hybrid is the cheap/fast lane: sentence-splitting in
# code (no extraction LLM) + Jev per-sentence verdicts
# (no classification LLM).

hybrid_contextual_recall = ContextualRecallMetric(
    threshold=0.7,
    include_reason=True,
    eval_mode="hybrid",
    # No `model=` required for the hybrid verdict path,
    # but you can still pass one if you want the LLM to
    # backstop any failed Jev calls.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Two expected sentences, both attributable — perfect
# recall. With `eval_mode="hybrid"`, the metric splits
# `expected_output` into sentences locally and asks Jev
# per-sentence.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output=(
        "You are eligible for a 30 day full refund at no extra cost. "
        "You can initiate a return from your order page."
    ),
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Returns can be initiated from the customer's order page in under a minute.",
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reports how confident the system_one
# model was on average across its calls — useful for
# debugging borderline cases.

hybrid_contextual_recall.measure(test_case)
print("---")
print(f"Score:      {hybrid_contextual_recall.score}")
print(f"Confidence: {hybrid_contextual_recall.confidence}")
print(f"Reason:     {hybrid_contextual_recall.reason}")
print(f"Passed:     {hybrid_contextual_recall.is_successful()}")