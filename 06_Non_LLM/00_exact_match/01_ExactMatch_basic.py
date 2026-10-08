from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# ExactMatchMetric checks whether `actual_output` is
# character-for-character equal to `expected_output`.
# It is purely string-level — no LLM, no API key, no
# token cost, zero latency from model calls.
#
# Score formula:
#   1  if actual_output == expected_output
#   0  otherwise
#
# Default `threshold=1.0` makes sense: matching is
# binary, so the only sensible "minimum passing" is
# "perfect match." Use this metric for classification
# labels, enum values, and canned responses where
# there's exactly one acceptable answer.

# ==========================================
# 2. THE METRIC INSTANCE
# ==========================================

exact = ExactMatchMetric(
    threshold=1.0,   # default; 1.0 == "must match exactly"
    verbose_mode=False,
)

# ==========================================
# 3. THE TEST CASE — perfect match
# ==========================================
# The model was asked to classify sentiment and
# returned "positive" — the same string in
# `expected_output`. The metric should score 1.0.

test_case = LLMTestCase(
    input="Classify the sentiment of: 'I love this!'",
    actual_output="positive",
    expected_output="positive",
)

# ==========================================
# 4. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and
# writes a report (Confident AI if configured, else
# local cache).

evaluate(test_cases=[test_case], metrics=[exact])