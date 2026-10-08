from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric

# ==========================================
# 1. THE METRIC (verbose mode)
# ==========================================
# `verbose_mode=True` prints the two strings the
# metric is comparing. This is the fastest way to
# diagnose a "looks correct but scores 0" result,
# because ExactMatchMetric is sensitive to:
#   - casing     ("Yes" != "yes")
#   - whitespace (" yes" != "yes", "yes " != "yes")
#   - newlines   ("yes\n" != "yes")
#   - smart quotes ("Yes" with curly quote != "Yes")
#   - accents    ("café" != "cafe")
# There is no normalization step.

exact = ExactMatchMetric(
    threshold=1.0,
    verbose_mode=True,   # prints actual vs expected
)

# ==========================================
# 2. STANDALONE — measure() and inspect
# ==========================================
# Bypass `evaluate()` to read the score and reason
# directly off the metric instance.

match_case = LLMTestCase(
    input="Translate 'yes' to French",
    actual_output="oui",
    expected_output="oui",
)
exact.measure(match_case)
print("--- match ---")
print(f"Score:   {exact.score}")           # 1.0
print(f"Reason:  {exact.reason}")          # "Outputs match exactly"
print(f"Passed:  {exact.is_successful()}")  # True

# ==========================================
# 3. STANDALONE — silent mismatch
# ==========================================
# Same word, different casing. ExactMatchMetric
# treats "Oui" and "oui" as different strings.

mismatch_case = LLMTestCase(
    input="Translate 'yes' to French",
    actual_output="Oui",       # capital O
    expected_output="oui",     # lowercase o
)
exact.measure(mismatch_case)
print("--- casing mismatch ---")
print(f"Score:   {exact.score}")            # 0.0
print(f"Reason:  {exact.reason}")
print(f"Passed:  {exact.is_successful()}")  # False