from deepeval.test_case import LLMTestCase
from deepeval.metrics import PatternMatchMetric

# ==========================================
# 1. THE METRIC (verbose mode)
# ==========================================
# `verbose_mode=True` prints the pattern, the input,
# and the regex result, so you can see exactly which
# anchor or character class failed. Useful while
# iterating on the regex itself.

us_phone = PatternMatchMetric(
    # North American phone: (555) 123-4567 or 555-123-4567
    pattern=(
        r"^(\(\d{3}\)\s|\d{3}-)"   # area code with ( ) or -
        r"\d{3}-\d{4}$"            # rest of number
    ),
    threshold=1.0,
    verbose_mode=True,
)

# ==========================================
# 2. STANDALONE — measure() and inspect
# ==========================================
# Bypass `evaluate()` to read the score and reason
# directly off the metric instance.

match_case = LLMTestCase(
    input="Generate a US phone number for support",
    actual_output="(555) 123-4567",
)
us_phone.measure(match_case)
print("--- match ---")
print(f"Score:   {us_phone.score}")           # 1.0
print(f"Reason:  {us_phone.reason}")
print(f"Passed:  {us_phone.is_successful()}")  # True

# ==========================================
# 3. STANDALONE — wrong format
# ==========================================
# Missing parens AND not using the hyphenator from
# the regex — neither alternation matches.

bad_case = LLMTestCase(
    input="Generate a US phone number for support",
    actual_output="555 123 4567",  # spaces, not dashes
)
us_phone.measure(bad_case)
print("--- wrong format ---")
print(f"Score:   {us_phone.score}")            # 0.0
print(f"Reason:  {us_phone.reason}")
print(f"Passed:  {us_phone.is_successful()}")  # False