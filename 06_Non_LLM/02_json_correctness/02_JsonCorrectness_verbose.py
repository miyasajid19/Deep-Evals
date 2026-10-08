from pydantic import BaseModel
from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import JsonCorrectnessMetric

# ==========================================
# 1. THE METRIC (verbose mode)
# ==========================================
# `verbose_mode=True` prints the schema, the
# attempted parse, and the validation result on
# every `measure()` call. Use it to debug why an
# output fails — was it unparseable JSON, or did it
# parse but with wrong types / missing fields?

class UserProfile(BaseModel):
    user_id: int
    email: str
    is_active: bool

profile_metric = JsonCorrectnessMetric(
    expected_schema=UserProfile,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. STANDALONE — measure() and inspect
# ==========================================
# Bypass `evaluate()` to read the score and reason
# directly off the metric instance.

valid_case = LLMTestCase(
    input="Create a user profile JSON",
    actual_output=(
        '{"user_id": 42, "email": "alice@example.com", '
        '"is_active": true}'
    ),
)
profile_metric.measure(valid_case)
print("--- valid ---")
print(f"Score:   {profile_metric.score}")           # 1.0
print(f"Reason:  {profile_metric.reason}")
print(f"Passed:  {profile_metric.is_successful()}")  # True

# ==========================================
# 3. STANDALONE — wrong field type
# ==========================================
# `user_id` should be an int; the LLM produced a
# string. The JSON parses but `pydantic` rejects
# the value. The reason will explain the failure
# (the LLM is asked to summarize the validation).

bad_type_case = LLMTestCase(
    input="Create a user profile JSON",
    actual_output=(
        '{"user_id": "forty-two", "email": "alice@example.com", '
        '"is_active": true}'
    ),
)
profile_metric.measure(bad_type_case)
print("--- wrong type ---")
print(f"Score:   {profile_metric.score}")            # 0.0
print(f"Reason:  {profile_metric.reason}")           # explains type error
print(f"Passed:  {profile_metric.is_successful()}")  # False