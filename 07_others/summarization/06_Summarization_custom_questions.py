from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

# ==========================================
# 1. THE METRIC (custom assessment_questions)
# ==========================================
# By default, the SummarizationMetric auto-generates
# `n` (default 5) close-ended yes/no questions from the
# input to compute the coverage score. Supply your own
# list when you already know what a good summary should
# answer — this pins the metric to the points you care
# about and removes the variability of auto-generated
# questions.
#
# Each question must be a closed-ended yes/no question
# that BOTH the input and the summary can answer. The
# coverage score is the fraction where both yield the
# same answer.

assessment_questions = [
    "Is the coverage score based on a percentage of 'yes' answers?",
    "Does a higher coverage score indicate a more comprehensive summary?",
    "Does the score check that the summary accurately represents the original?",
]

summarization = SummarizationMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    assessment_questions=assessment_questions,
)

# ==========================================
# 2. THE TEST CASE — full coverage
# ==========================================
# This summary answers all three custom questions the
# same way the original does. Expect a coverage score
# of 1.0 and (assuming the alignment score is also
# high) a final score of 1.0.

input_text = """
The 'coverage score' is calculated as the percentage of assessment questions
for which both the summary and the original document provide a 'yes' answer. This
method ensures that the summary not only includes key information from the original
text but also accurately represents it. A higher coverage score indicates a
more comprehensive and faithful summary, signifying that the summary effectively
encapsulates the crucial points and details from the original content.
"""

actual_output = """
The coverage score is a percentage-based metric that reflects how faithfully a
summary captures and represents the key points of the original document. A
higher score indicates a more comprehensive summary.
"""

test_case = LLMTestCase(
    input=input_text,
    actual_output=actual_output,
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# With three custom questions and full coverage, expect
# score 1.0. Drop the summary down to one sentence
# about the average that uses percentage-style framing
# and the coverage score stays high; drop the
# 'higher score' claim and coverage falls.

summarization.measure(test_case)
print("---")
print(f"Score:  {summarization.score}")
print(f"Reason: {summarization.reason}")
print(f"Passed: {summarization.is_successful()}")