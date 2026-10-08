from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

# ==========================================
# 1. THE METRIC — show score_breakdown
# ==========================================
# `metric.score_breakdown` exposes the underlying
# alignment score and coverage score separately after
# the metric runs. The final score is the MIN of the
# two, so checking the breakdown is the fastest way to
# see which half is dragging it down — a summary that's
# accurate but drops key points is capped by a low
# coverage score; a summary that hallucinates is capped
# by a low alignment score.

summarization = SummarizationMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. CASE A — perfect alignment AND coverage
# ==========================================
# Both halves hit 1.0, so the min is 1.0.

perfect_input = """
Venus is the second planet from the Sun. It's the hottest planet in the
solar system due to its thick CO2 atmosphere that traps heat via a runaway
greenhouse effect.
"""

perfect_output = """
Venus is the second planet from the Sun and is the hottest planet in the
solar system, made so by a thick CO2 atmosphere that traps heat in a runaway
greenhouse effect.
"""

perfect_case = LLMTestCase(
    input=perfect_input,
    actual_output=perfect_output,
)

summarization.measure(perfect_case)
print("--- perfect run ---")
print(f"Score:    {summarization.score}")
print(f"Breakdown: {summarization.score_breakdown}")
print(f"Passed:   {summarization.is_successful()}")

# ==========================================
# 3. CASE B — factually wrong (alignment low)
# ==========================================
# The summary invents a moon — alignment drops, coverage
# stays high. Final score = min(low alignment, high
# coverage) = low alignment.

wrong_input = """
Jupiter is the largest planet in the solar system. It has at least 95 known
moons, including the four large Galilean moons.
"""

wrong_output = """
Jupiter is the largest planet in the solar system. It has exactly four
moons, the four Galilean moons discovered by Galileo himself.
"""

wrong_case = LLMTestCase(
    input=wrong_input,
    actual_output=wrong_output,
)

summarization.measure(wrong_case)
print("--- invented fact ---")
print(f"Score:    {summarization.score}")
print(f"Breakdown: {summarization.score_breakdown}")
print(f"Passed:   {summarization.is_successful()}")

# ==========================================
# 4. CASE C — accurate but incomplete (coverage low)
# ==========================================
# The summary is fully aligned with the input but drops
# the moon count and the Galilean moons. Coverage drops,
# alignment stays high. Final = min(high alignment,
# low coverage) = low coverage.

incomplete_input = wrong_input

incomplete_output = """
Jupiter is the largest planet in the solar system and has many known moons.
"""

incomplete_case = LLMTestCase(
    input=incomplete_input,
    actual_output=incomplete_output,
)

summarization.measure(incomplete_case)
print("--- dropped details ---")
print(f"Score:    {summarization.score}")
print(f"Breakdown: {summarization.score_breakdown}")
print(f"Passed:   {summarization.is_successful()}")