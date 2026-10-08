from pydantic import BaseModel
from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import JsonCorrectnessMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses scoring to binary
# (0 or 1) AND overrides `threshold` to 1, so
# `is_successful()` matches the score exactly.
# Useful when ANY schema deviation is unacceptable
# — e.g. compliance, contract validation, or
# downstream tools that depend on field presence.
#
# For JsonCorrectnessMetric this is largely
# cosmetic: the score is already binary because
# schema validation either passes or fails. But
# strict_mode is the documented way to declare
# "any failure means the test fails."

class Order(BaseModel):
    order_id: str
    quantity: int
    total: float

strict_json = JsonCorrectnessMetric(
    expected_schema=Order,
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. VALID JSON — score 1
# ==========================================

valid_case = LLMTestCase(
    input="Confirm the order",
    actual_output=(
        '{"order_id": "ORD-001", "quantity": 3, "total": 49.95}'
    ),
)
strict_json.measure(valid_case)
print("--- valid ---")
print(f"Score:   {strict_json.score}")           # 1.0
print(f"Reason:  {strict_json.reason}")
print(f"Passed:  {strict_json.is_successful()}")  # True

# ==========================================
# 3. MISSING REQUIRED FIELD — score 0
# ==========================================
# `total` is missing entirely. Strict mode flags it
# as a hard fail.

missing_field_case = LLMTestCase(
    input="Confirm the order",
    actual_output=(
        '{"order_id": "ORD-001", "quantity": 3}'
    ),
)
strict_json.measure(missing_field_case)
print("--- missing field ---")
print(f"Score:   {strict_json.score}")            # 0.0
print(f"Reason:  {strict_json.reason}")
print(f"Passed:  {strict_json.is_successful()}")  # False

# ==========================================
# 4. EXTRA FIELDS — also fails in strict mode
# ==========================================
# pydantic's default is "ignore extra fields"
# rather than reject them, so this typically passes.
# To reject extras, configure `model_config =
# ConfigDict(extra="forbid")` on your schema.

extra_case = LLMTestCase(
    input="Confirm the order",
    actual_output=(
        '{"order_id": "ORD-001", "quantity": 3, "total": 49.95, '
        '"discount": "ten-percent"}'
    ),
)
strict_json.measure(extra_case)
print("--- extra field (default config) ---")
print(f"Score:   {strict_json.score}")
print(f"Reason:  {strict_json.reason}")