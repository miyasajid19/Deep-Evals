from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# SummarizationMetric checks whether your LLM's summary
# is *aligned* with the original text (no hallucinations
# or contradictions) AND *covers* its key points.
#
# Two scores combine into the final value:
#   - Alignment score: fraction of summary claims that
#     agree with the input. Equivalent to a Hallucination
#     check.
#   - Coverage score:  fraction of generated yes/no
#     questions about the input that BOTH the input and
#     the summary answer the same.
#
#   Summarization = min(Alignment, Coverage)
#
# Note: SummarizationMetric is the only default deepeval
# metric that is NOT cacheable, because the coverage
# score generates fresh assessment questions each run.
#
# `assessment_questions` is OPTIONAL — leave it unset and
# the metric auto-generates `n` (default 5) close-ended
# yes/no questions about the input. Supply your own when
# you know exactly what a good summary should answer.
#
# This metric is `referenceless`: only `input` and
# `actual_output` are needed — no `expected_output`.

summarization = SummarizationMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Input is the original text; actual_output is what
# your LLM produced. The summary here captures the
# key claim (coverage score should be high) without
# inventing detail (alignment should also be high).

input_text = """
The 'coverage score' is calculated as the percentage of assessment questions
for which both the summary and the original document provide a 'yes' answer. This
method ensures that the summary not only includes key information from the original
text but also accurately represents it. A higher coverage score indicates a
more comprehensive and faithful summary, signifying that the summary effectively
encapsulates the crucial points and details from the original content.
"""

actual_output = """
The coverage score quantifies how well a summary captures and
accurately represents key information from the original text,
with a higher score indicating greater comprehensiveness.
"""

test_case = LLMTestCase(
    input=input_text,
    actual_output=actual_output,
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[test_case], metrics=[summarization])