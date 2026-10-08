from deepeval.test_case import LLMTestCase
from deepeval.metrics import PatternMatchMetric

# ==========================================
# 1. THE METRIC — case-insensitive matching
# ==========================================
# PatternMatchMetric defaults to `ignore_case=False`.
# For patterns where casing is irrelevant — state
# codes, country codes, free-form tokens — set
# `ignore_case=True`. The metric uses the
# `re.IGNORECASE` flag, so letter classes
# `[A-Z]` match both upper and lower case.

# ==========================================
# 2. TWO METRICS, ONE PATTERN
# ==========================================
# Same regex `^(yes|no|maybe)$`. The case-sensitive
# one fails on "YES"; the case-insensitive one
# passes. The point: case-sensitivity is a single
# boolean, not something the regex itself fixes.

yes_no_pattern = r"^(yes|no|maybe)$"

case_sensitive = PatternMatchMetric(
    pattern=yes_no_pattern,
    ignore_case=False,    # default
    threshold=1.0,
)

case_insensitive = PatternMatchMetric(
    pattern=yes_no_pattern,
    ignore_case=True,
    threshold=1.0,
)

# ==========================================
# 3. SAME OUTPUT, TWO OUTCOMES
# ==========================================

mixed_case = LLMTestCase(
    input="Did the user confirm?",
    actual_output="YES",   # all caps
)

case_sensitive.measure(mixed_case)
print("--- case sensitive on 'YES' ---")
print(f"Score:   {case_sensitive.score}")           # 0.0
print(f"Reason:  {case_sensitive.reason}")

case_insensitive.measure(mixed_case)
print("--- case insensitive on 'YES' ---")
print(f"Score:   {case_insensitive.score}")           # 1.0
print(f"Reason:  {case_insensitive.reason}")
print(f"Passed:  {case_insensitive.is_successful()}")  # True

# ==========================================
# 4. WHEN TO USE ignore_case
# ==========================================
# Use `ignore_case=True` when:
#   - the model is free to vary casing
#   - the symbol set is conventional (US state codes)
#   - you want to focus the metric on shape, not
#     spelling (e.g. "API keys are uppercase" is not
#     a property you want this metric to enforce)
# Keep `ignore_case=False` when casing carries
# meaning: hex codes, enum values, product SKUs.