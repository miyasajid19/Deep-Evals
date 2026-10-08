from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps the
# LLM goes through:
#   - the factual `truths` extracted from the input
#   - the `claims` extracted from the actual_output
#   - the per-claim alignment verdict (aligned vs
#     contradicts)
#   - the auto-generated `assessment_questions`
#   - the per-question coverage verdict (same answer
#     from input and summary?)
#
# Verbose mode is the fastest way to debug "why is my
# coverage score low" — you can see exactly which
# assessment question the summary failed.

summarization = SummarizationMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — perfect run
# ==========================================
# A complete, accurate summary should hit high alignment
# AND high coverage. Expect score 1.0.

input_text = """
The Eiffel Tower, completed in 1889, stands 330 metres tall in Paris. It was
built for the 1889 World's Fair and remains one of the most-visited paid
monuments in the world, drawing roughly 7 million visitors each year.
"""

actual_output = """
The Eiffel Tower, completed in 1889 in Paris, stands 330 metres tall and
attracts about 7 million visitors a year, making it one of the most
visited paid monuments in the world.
"""

test_case = LLMTestCase(
    input=input_text,
    actual_output=actual_output,
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason.
# Verbose output appears above.

summarization.measure(test_case)
print("---")
print(f"Score:   {summarization.score}")
print(f"Reason:  {summarization.reason}")
print(f"Passed:  {summarization.is_successful()}")