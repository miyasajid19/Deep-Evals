from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

# ==========================================
# 1. THE METRIC (truths_extraction_limit)
# ==========================================
# The SummarizationMetric extracts an unbounded set of
# factual `truths` from the input. For long inputs, that
# can balloon the alignment cost — and penalize the
# summary for missing minor, low-importance points.
#
# `truths_extraction_limit` caps how many truths are
# pulled out. Truths are ordered by importance (decided
# by the judge LLM), so a limit prioritizes the most
# important facts. This is the right knob when:
#   - the input is a long document and most facts are
#     minor,
#   - you've already established what a good summary
#     looks like and don't want to over-penalize for
#     fine-grained omissions,
#   - you want faster evaluations.
#
# `n` (default 5) controls assessment questions; this
# controls alignment truths. Different parameters.

summarization = SummarizationMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    truths_extraction_limit=3,  # only consider top-3 most important truths
    n=3,                        # and 3 assessment questions for coverage
)

# ==========================================
# 2. THE TEST CASE — long input, brief summary
# ==========================================
# The input lists six facts (a–f). The summary hits the
# three MOST important (a, b, c). With
# truths_extraction_limit=3 the metric only judges
# against the top three, so the brief summary scores
# perfectly even though it omits d, e, f.

input_text = """
(a) The Hubble Space Telescope was launched in 1990.
(b) It orbits about 540 km above Earth's surface.
(c) Hubble has revolutionized our view of distant galaxies.
(d) Its primary mirror is 2.4 metres across.
(e) Hubble has been serviced five times by Space Shuttle crews.
(f) NASA expects Hubble to remain operational through the late 2020s.
"""

actual_output = """
The Hubble Space Telescope, launched in 1990 and orbiting about 540 km above
Earth, has given astronomers a revolutionary view of distant galaxies.
"""

test_case = LLMTestCase(
    input=input_text,
    actual_output=actual_output,
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================

summarization.measure(test_case)
print("---")
print(f"Score:  {summarization.score}")
print(f"Reason: {summarization.reason}")
print(f"Passed: {summarization.is_successful()}")