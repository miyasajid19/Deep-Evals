from pydantic import BaseModel
from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import JsonCorrectnessMetric

# ==========================================
# 1. THE METRIC — system_one eval mode
# ==========================================
# JsonCorrectnessMetric supports three eval modes
# for how the FAILURE REASON is generated:
#
#   - "llm"        (default) ask an LLM to summarize
#                   the pydantic ValidationError into
#                   a human-readable sentence.
#   - "hybrid"     LLM extract + deterministic check
#   - "system_one" bypass the LLM and use the raw
#                   ValidationError text as the reason.
#
# `system_one` is the right choice when:
#   - you want zero token cost even when outputs fail
#   - the ValidationError is already informative
#     (it names the field and the offending value)
#   - you want the metric to be fully deterministic
#     end-to-end
#
# IMPORTANT: scoring is identical across all three
# modes. The eval_mode only changes the FAILURE
# REASON — pass fail cases have an empty reason.

class Order(BaseModel):
    order_id: str
    quantity: int

system_one_metric = JsonCorrectnessMetric(
    expected_schema=Order,
    model=CustomOpenAI(),       # ignored in system_one mode
    include_reason=True,
    eval_mode="system_one",     # no LLM call on failure
    verbose_mode=True,
)

# ==========================================
# 2. VALID OUTPUT — score 1
# ==========================================

valid_case = LLMTestCase(
    input="Confirm the order",
    actual_output='{"order_id": "ORD-001", "quantity": 3}',
)
system_one_metric.measure(valid_case)
print("--- valid ---")
print(f"Score:   {system_one_metric.score}")           # 1.0
print(f"Reason:  {system_one_metric.reason!r}")         # "" (empty)
print(f"Passed:  {system_one_metric.is_successful()}")  # True

# ==========================================
# 3. INVALID OUTPUT — score 0, no LLM invoked
# ==========================================
# `quantity` is a string, not an int. system_one
# surfaces the raw ValidationError: the model is
# not called, so this incurs zero token cost.

invalid_case = LLMTestCase(
    input="Confirm the order",
    actual_output='{"order_id": "ORD-001", "quantity": "three"}',
)
system_one_metric.measure(invalid_case)
print("--- invalid ---")
print(f"Score:   {system_one_metric.score}")            # 0.0
print(f"Reason:  {system_one_metric.reason}")           # raw validation error
print(f"Passed:  {system_one_metric.is_successful()}")  # False

# ==========================================
# 4. WHEN eval_mode MATTERS
# ==========================================
# Compare:
#   - `eval_mode="llm"`: reason is a polished
#     sentence ("Expected int for 'quantity' but got
#     a string"). Costs ~100 tokens per failure.
#   - `eval_mode="system_one"`: reason is the raw
#     pydantic message. Costs 0 tokens.
# For CI pipelines that fail thousands of cases
# per run, system_one keeps the bill at zero.
# For human-facing reports, llm gives a friendlier
# summary. The score is identical either way.