from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Summarization: the LLM still does the heavy
# lifting — extracting `truths` from the input AND
# extracting `claims` from the actual_output AND
# generating the `assessment_questions` for the coverage
# score. Only the alignment classification is delegated
# to Jev (one yes/no question per claim, with
# `P(yes) > 0.65` counted as aligned, `< 0.35` as
# contradictory, anything in between as borderline).
#
# The coverage score stays on the LLM — only alignment
# changes judges. The equation is identical to `llm`
# mode:
#   Summarization = min(Alignment, Coverage)
#
# If a Jev call fails, the metric silently defers that
# alignment decision to the LLM.

hybrid_summarization = SummarizationMetric(
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still extracts truths/claims/questions
    include_reason=True,
    eval_mode="hybrid",         # Jev classifies each claim alignment
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same clean summary as the basic run — alignment AND
# coverage should both be high. Expect score 1.0.

input_text = """
Photosynthesis is the process used by plants to convert light energy into
chemical energy stored in glucose. It releases oxygen as a byproduct and
takes place primarily in the leaves.
"""

actual_output = """
Plants use photosynthesis to turn sunlight into chemical energy (glucose) and
release oxygen as a byproduct, mainly in their leaves.
"""

test_case = LLMTestCase(
    input=input_text,
    actual_output=actual_output,
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# `metric.confidence` reports how confident Jev was on
# average across its per-claim calls. The reason lists
# each claim's verdict (aligned / contradictory /
# borderline) with its probability.

hybrid_summarization.measure(test_case)
print("---")
print(f"Score:      {hybrid_summarization.score}")
print(f"Confidence: {hybrid_summarization.confidence}")
print(f"Reason:     {hybrid_summarization.reason}")
print(f"Passed:     {hybrid_summarization.is_successful()}")