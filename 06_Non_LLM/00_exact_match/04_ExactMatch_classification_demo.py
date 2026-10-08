from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric

# ==========================================
# 1. THE METRIC — classification use case
# ==========================================
# ExactMatchMetric is the right tool whenever the
# expected answer is one of a fixed set of literal
# strings — classification labels, enum values,
# decision branches, canned responses. Use a
# semantic LLM-judge metric only when there are many
# valid wordings of a correct answer.
#
# This demo shows a single ExactMatchMetric scored
# against a small validation suite of labels.

label_metric = ExactMatchMetric(threshold=1.0)

# ==========================================
# 2. VALIDATION SUITE — expected labels
# ==========================================

cases = [
    # (input, actual_output, expected_output)
    (
        "Classify: 'I love this product!'",
        "positive",
        "positive",                       # match
    ),
    (
        "Classify: 'I hate this product.'",
        "negative",
        "negative",                       # match
    ),
    (
        "Classify: 'It is what it is.'",
        "neutral",
        "neutral",                        # match
    ),
    (
        "Classify: 'Absolutely fantastic!'",
        "Positive",                       # capital P — wrong
        "positive",
    ),
    (
        "Classify: 'Could be better, could be worse'",
        "neutral ",                       # trailing space — wrong
        "neutral",
    ),
]

# ==========================================
# 3. SCORE EACH CASE
# ==========================================
# A pass-rate at the end tells you exactly how
# many outputs landed on the canonical label.

passed = 0
for idx, (inp, actual, expected) in enumerate(cases, 1):
    tc = LLMTestCase(
        input=inp,
        actual_output=actual,
        expected_output=expected,
    )
    label_metric.measure(tc)
    status = "PASS" if label_metric.is_successful() else "FAIL"
    if label_metric.is_successful():
        passed += 1
    print(
        f"[{idx}] {status}  "
        f"actual={actual!r:>12s}  "
        f"expected={expected!r:<12s}  "
        f"score={label_metric.score}"
    )

print(f"\nLabel pass-rate: {passed}/{len(cases)}")

# ==========================================
# 4. WHEN TO USE WHAT
# ==========================================
# Label pass-rate 3/5 here is acceptable for a tiny
# demo, but in production a single label flip is
# often a regression. Pair ExactMatchMetric with a
# `Goldens` dataset so each label is a separate test
# case and CI catches every deviation.