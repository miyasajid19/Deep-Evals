from pydantic import BaseModel
from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import JsonCorrectnessMetric

# ==========================================
# 1. THE METRIC — failure-mode showcase
# ==========================================
# JsonCorrectnessMetric scores 0 whenever pydantic
# cannot validate `actual_output` against
# `expected_schema`. The five canonical ways an
# LLM-generated JSON can fail:
#
#   1. malformed JSON            (parse error)
#   2. valid JSON, wrong shape   (extra / missing field)
#   3. valid JSON, wrong types   (string vs int)
#   4. nested object mismatch    (deep field fails)
#   5. silently null field       (LLM gave up)
#
# All five should score 0 here. Pair this demo
# with a Goldens dataset to keep each failure
# pattern as its own test case in CI.

class Recipe(BaseModel):
    name: str
    servings: int
    prep_time_minutes: int

failure_metric = JsonCorrectnessMetric(
    expected_schema=Recipe,
    model=CustomOpenAI(),
    include_reason=True,
    eval_mode="system_one",     # cheap, deterministic reason
    verbose_mode=False,
)

# ==========================================
# 2. THE FIVE FAILURES
# ==========================================

failures = [
    ("malformed JSON",
     "Sorry, here's the recipe: name=Pasta, servings=4"),

    ("missing field",
     '{"name": "Pasta", "servings": 4}'),

    ("wrong type (int → str)",
     '{"name": "Pasta", "servings": "four", "prep_time_minutes": 20}'),

    ("nested field absent (flat shape here, but field is null)",
     '{"name": null, "servings": 4, "prep_time_minutes": 20}'),

    ("LLM gave up — empty string",
     ""),
]

# ==========================================
# 3. RUN EACH FAILURE
# ==========================================
# Every entry below should score 0. If any scores 1,
# the schema is too permissive (or you meant to
# test the "all-pass" case).

unexpected_passes = 0
for label, output in failures:
    tc = LLMTestCase(
        input="Give me a JSON recipe for pasta",
        actual_output=output,
    )
    failure_metric.measure(tc)
    passed = failure_metric.is_successful()
    if passed:
        unexpected_passes += 1
    print(
        f"{label:<45s}  score={failure_metric.score}  "
        f"reason={failure_metric.reason}"
    )

print(f"\nUnexpected passes: {unexpected_passes}/{len(failures)}")
# Expected: 0

# ==========================================
# 4. WHEN TO ESCALATE FROM TOOLING TO AN LLM JUDGE
# ==========================================
# JsonCorrectnessMetric checks SHAPE, not semantics.
# A perfectly shaped JSON with a wrong value still
# scores 1 (it parsed, types match). To judge the
# contents — "is the recipe actually correct?" —
# pair this metric with an LLM-judge metric such
# as `GEval` or `AnswerRelevancy` over the same
# `LLMTestCase`.