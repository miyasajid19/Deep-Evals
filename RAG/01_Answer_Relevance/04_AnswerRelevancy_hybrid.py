from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid keeps the LLM for the statement-extraction step
# but hands the per-statement classification to a System
# One model (Jev). Jev answers one binary question per
# statement; the metric aggregates them using the same
# (# relevant / # total) equation as the pure LLM mode.
#
# Hybrid is the cheap/fast lane: it's cheaper than running
# a full LLM per statement, more reliable than running
# Jev on the whole answer in one shot (system_one mode).

hybrid_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=CustomOpenAI(),       # LLM still does extraction
    include_reason=True,
    eval_mode="hybrid",         # Jev classifies, not the LLM
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same Q&A as the basic run — every statement is relevant.
# With `eval_mode="hybrid"`, the LLM extracts statements,
# Jev judges each one. If a Jev call fails, the metric
# silently defers that decision to the LLM.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "We offer a 30-day full refund at no extra cost. "
        "You can initiate a return from your order page."
    ),
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reports how confident the system_one model
# was on average across its calls — useful for debugging
# borderline cases.

hybrid_relevancy.measure(test_case)
print("---")
print(f"Score:      {hybrid_relevancy.score}")
print(f"Confidence: {hybrid_relevancy.confidence}")
print(f"Reason:     {hybrid_relevancy.reason}")
print(f"Passed:     {hybrid_relevancy.is_successful()}")