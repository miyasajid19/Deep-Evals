from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if the summary is perfectly faithful AND fully
#       covers every point in the input
#   - 0 if anything is missed or contradicted
# It also overrides `threshold` to 1, so
# `is_successful()` matches the binary score exactly.
#
# Use it when even minor drift is unacceptable — e.g.
# compliance/legal summaries where any fabrication or
# omission is a hard fail.

strict_summarization = SummarizationMetric(
    threshold=0.5,             # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. PERFECT RUN — score 1
# ==========================================
# The summary preserves every fact and adds nothing.

input_text = """
Mars is the fourth planet from the Sun. It has a thin atmosphere of mostly
carbon dioxide and two small moons, Phobos and Deimos.
"""

perfect_output = """
Mars is the fourth planet from the Sun, has a thin CO2 atmosphere, and two
moons called Phobos and Deimos.
"""

perfect_case = LLMTestCase(
    input=input_text,
    actual_output=perfect_output,
)

strict_summarization.measure(perfect_case)
print("--- perfect summary ---")
print(f"Score:   {strict_summarization.score}")          # 1.0
print(f"Reason:  {strict_summarization.reason}")
print(f"Passed:  {strict_summarization.is_successful()}")  # True

# ==========================================
# 3. ONE FACT WRONG — score 0
# ==========================================
# The summary invents "Olympus Mons" — a fact absent from
# the input. Strict mode drops the score to 0.

wrong_fact_output = """
Mars is the fourth planet from the Sun and has a thin atmosphere. Its tallest
mountain is Olympus Mons, a massive volcano.
"""

wrong_fact_case = LLMTestCase(
    input=input_text,
    actual_output=wrong_fact_output,
)

strict_summarization.measure(wrong_fact_case)
print("--- invented fact ---")
print(f"Score:   {strict_summarization.score}")          # 0.0
print(f"Reason:  {strict_summarization.reason}")
print(f"Passed:  {strict_summarization.is_successful()}")  # False

# ==========================================
# 4. ACCURATE BUT INCOMPLETE — score 0
# ==========================================
# The summary is factually correct, but drops the two
# moons. Coverage breaks, and because the score is the
# MIN of alignment and coverage, the final score is 0
# even though the alignment score would be high.

incomplete_output = """
Mars is the fourth planet from the Sun and has a thin atmosphere of mostly
carbon dioxide.
"""

incomplete_case = LLMTestCase(
    input=input_text,
    actual_output=incomplete_output,
)

strict_summarization.measure(incomplete_case)
print("--- dropped the moons ---")
print(f"Score:   {strict_summarization.score}")          # 0.0
print(f"Reason:  {strict_summarization.reason}")
print(f"Passed:  {strict_summarization.is_successful()}")  # False