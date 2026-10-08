from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# ExactMatchMetric is already binary — the score
# can only be 0 or 1 — so `strict_mode` is a
# no-op in practice. It exists to be consistent
# with the rest of the deepeval metric family:
# any metric can be flipped into a 1-or-0 gate.
#
# When `strict_mode=True`:
#   - threshold is overridden to 1
#   - score stays 0 or 1
# For ExactMatchMetric this changes nothing about
# scoring, but it's still useful as documentation
# that "any deviation is unacceptable."

strict_exact = ExactMatchMetric(
    threshold=0.5,        # overridden to 1 in strict mode
    strict_mode=True,
)

# ==========================================
# 2. MATCHING OUTPUT — score 1
# ==========================================

match_case = LLMTestCase(
    input="What is 2 + 2?",
    actual_output="4",
    expected_output="4",
)
strict_exact.measure(match_case)
print("--- match ---")
print(f"Score:   {strict_exact.score}")           # 1.0
print(f"Passed:  {strict_exact.is_successful()}")  # True
print(f"Reason:  {strict_exact.reason}")

# ==========================================
# 3. WRONG ANSWER — score 0
# ==========================================
# Even a single character difference is a hard fail.

wrong_case = LLMTestCase(
    input="What is 2 + 2?",
    actual_output="5",
    expected_output="4",
)
strict_exact.measure(wrong_case)
print("--- wrong answer ---")
print(f"Score:   {strict_exact.score}")            # 0.0
print(f"Passed:  {strict_exact.is_successful()}")  # False
print(f"Reason:  {strict_exact.reason}")

# ==========================================
# 4. TRAILING WHITESPACE — also fails
# ==========================================
# "4 " (with a trailing space) is not equal to "4".
# Set verbose_mode=True to see the exact bytes.

trailing_ws_case = LLMTestCase(
    input="What is 2 + 2?",
    actual_output="4 ",    # trailing space
    expected_output="4",
)
strict_exact.measure(trailing_ws_case)
print("--- trailing whitespace ---")
print(f"Score:   {strict_exact.score}")            # 0.0
print(f"Reason:  {strict_exact.reason}")