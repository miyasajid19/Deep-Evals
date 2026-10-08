from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric

# ==========================================
# 1. THE METRIC — pitfall showcase
# ==========================================
# ExactMatchMetric is the strictest comparison in
# deepeval. Every byte must align. This script walks
# through the seven most common reasons an output
# "looks correct" but scores 0:
#
#   1. different casing
#   2. leading / trailing whitespace
#   3. internal extra space
#   4. trailing newline
#   5. smart / curly quotes
#   6. accented vs ASCII characters
#   7. a stray period
#
# Run this once to internalize what fails, then
# strip / normalize upstream if any of these matter
# to you (or switch to a semantic metric).

exact = ExactMatchMetric(threshold=1.0, verbose_mode=True)

CANON = "Paris"   # the single acceptable answer

pitfalls = [
    ("different casing",         "paris",     "Paris"),
    ("leading whitespace",       " Paris",    "Paris"),
    ("trailing whitespace",      "Paris ",    "Paris"),
    ("internal extra space",     "Par is",    "Paris"),
    ("trailing newline",         "Paris\n",   "Paris"),
    ("smart curly quotes",       "“Paris”", "Paris"),
    ("accented character",       "Pàris",      "Paris"),
    ("extra period",             "Paris.",     "Paris"),
]

# ==========================================
# 2. RUN EACH PITFALL
# ==========================================
# Every entry below should score 0 — they all differ
# from `CANON` by a single invisible or trivial edit.

passed_unexpectedly = 0
for label, actual, expected in pitfalls:
    tc = LLMTestCase(
        input="Capital of France?",
        actual_output=actual,
        expected_output=expected,
    )
    exact.measure(tc)
    if exact.is_successful():
        passed_unexpectedly += 1
    print(f"{label:<22s}  score={exact.score}  reason={exact.reason}")

print(f"\nUnexpected passes: {passed_unexpectedly}/{len(pitfalls)}")
# Expected: 0 — every pitfall should score 0.

# ==========================================
# 3. WHEN YOU NEED FUZZY MATCHING
# ==========================================
# If your outputs frequently vary in casing or
# whitespace (e.g. the LLM adds a trailing period
# half the time), pick a metric that handles
# normalization or semantics:
#   - lower/upper + strip both strings before compare
#   - use a `CustomMetric` subclassing `BaseMetric`
#   - or switch to an LLM-judge metric
# ExactMatchMetric is intentionally unforgiving.