from deepeval.test_case import LLMTestCase
from deepeval.metrics import PatternMatchMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# PatternMatchMetric verifies that `actual_output`
# conforms to a provided regular expression pattern.
# It uses Python's built-in `re.fullmatch`, which
# means the pattern must match the ENTIRE output
# (anchored on both ends), not a substring inside it.
#
# Score formula:
#   1  if re.fullmatch(pattern, actual_output)
#   0  otherwise
#
# The metric is referenceless (no `expected_output`)
# and non-LLM (no model call, no token cost, no
# latency from model inference).

# ==========================================
# 2. THE METRIC INSTANCE — email pattern
# ==========================================
# A reasonable "good-enough" email regex:
#   ^[\w\.-]+@[\w\.-]+\.\w+$
#   local@domain.tld  with letters/digits/_/./- on
#   either side of the @, and at least one dot in
#   the domain.

email_pattern = PatternMatchMetric(
    pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
    ignore_case=False,    # case-sensitive by default
    threshold=1.0,
    verbose_mode=False,
)

# ==========================================
# 3. THE TEST CASE — well-formed email
# ==========================================
# The LLM was asked to generate an email address and
# produced a syntactically valid one. PatternMatch
# checks only the SHAPE — not whether the address
# actually exists or belongs to a real person.

test_case = LLMTestCase(
    input="Generate a valid email address.",
    actual_output="example.user@domain.com",
)

# ==========================================
# 4. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and
# writes a report (Confident AI if configured, else
# local cache).

evaluate(test_cases=[test_case], metrics=[email_pattern])