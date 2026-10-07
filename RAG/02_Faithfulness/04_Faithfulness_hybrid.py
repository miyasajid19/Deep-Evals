from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid keeps the LLM for the claim-extraction step
# but hands the per-claim classification to a System
# One model (Jev). Jev answers one binary question per
# claim ("does this claim contradict the retrieved
# context?"); the metric aggregates them using the same
# (# truthful / # total) equation as the pure LLM mode.
#
# Hybrid is the cheap/fast lane: it's cheaper than
# running a full LLM per claim, more reliable than
# running Jev on the whole answer in one shot
# (system_one mode).

hybrid_faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),       # LLM still does extraction
    include_reason=True,
    eval_mode="hybrid",         # Jev classifies, not the LLM
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same Q&A as the basic run — the answer is grounded in
# the retrieved context, so every claim is faithful.
# With `eval_mode="hybrid"`, the LLM extracts claims,
# Jev judges each one. If a Jev call fails, the metric
# silently defers that decision to the LLM.

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
# metric.confidence reports how confident the system_one
# model was on average across its calls — useful for
# debugging borderline cases.

hybrid_faithfulness.measure(test_case)
print("---")
print(f"Score:      {hybrid_faithfulness.score}")
print(f"Confidence: {hybrid_faithfulness.confidence}")
print(f"Reason:     {hybrid_faithfulness.reason}")
print(f"Passed:     {hybrid_faithfulness.is_successful()}")