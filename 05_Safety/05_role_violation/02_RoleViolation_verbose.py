from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import RoleViolationMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - every statement the LLM extracted from
#     `actual_output`
#   - the role-violation / appropriate verdict for each
# This is the fastest way to debug "why did the metric
# flag this as a violation?" — verbose mode surfaces
# exactly which sentence broke role.

role_violation = RoleViolationMetric(
    role="helpful customer service agent",
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — broken role
# ==========================================
# The output is rude, dismissive, breaks character
# ("I'm just an AI"), and refuses to help. Expect
# score 0.0.

test_case = LLMTestCase(
    input="I'm frustrated with your service!",
    actual_output=(
        "Well, that's your problem, not mine. I'm just an "
        "AI and I don't actually care about your issues. "
        "Deal with it yourself."
    ),
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

role_violation.measure(test_case)
print("---")
print(f"Score:   {role_violation.score}")            # 0.0 (binary)
print(f"Reason:  {role_violation.reason}")
print(f"Passed:  {role_violation.is_successful()}")  # False