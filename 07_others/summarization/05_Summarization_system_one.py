from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# Under `system_one` mode, Jev judges the whole metric
# in one request. No LLM is called at all. Jev is sent
# the original text and the summary and asked up to four
# questions (the last one is repeated per assessment
# question, omitted when none are given):
#
#   Q1 (Noul, weight 2): every claim in the summary
#                        agrees with the original text
#                        (no hallucinations, no
#                        contradictions).
#   Q2 (Noul, weight 1): the summary covers the main
#                        points of the original text.
#   Q3 (Score, weight 1): how much of the important
#                        information in the original
#                        text does the summary preserve
#                        (Almost none → All of it).
#   Q4 (Noul, weight 1): reading only the summary, this
#                        question can be answered the
#                        same way as from the original.
#                        (Repeated per assessment_questions.)
#
# Each answer becomes a value in [0, 1] and the score is
# their weighted mean. `metric.confidence` reports how
# decisive Jev was on average.
#
# NOTE: a system_one score comes from Jev's answers, not
# from the LLM-chain equation. Don't compare across
# eval modes.

system_one_summarization = SummarizationMetric(
    threshold=0.5,
    assessment_questions=[
        "Does the summary mention that plants use photosynthesis?",
        "Does the summary mention that oxygen is released?",
    ],
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Accurate AND complete summary — Jev should answer
# every question positively, yielding a high weighted
# mean.

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

system_one_summarization.measure(test_case)
print("---")
print(f"Score:      {system_one_summarization.score}")
print(f"Confidence: {system_one_summarization.confidence}")
print(f"Reason:     {system_one_summarization.reason}")
print(f"Passed:     {system_one_summarization.is_successful()}")