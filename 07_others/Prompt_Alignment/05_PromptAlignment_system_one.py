from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# Under `system_one` mode, Jev judges the whole metric
# in one request. No LLM is called at all. Jev is sent
# the raw `input` and `actual_output` and asked one
# question per instruction:
#
#   Q1 (Noul, weight 1): actual_output completely
#                        follows this prompt instruction
#                        when responding to input.
#   Q2 (Noul, weight 1): actual_output completely
#                        follows this prompt instruction
#                        when responding to input.
#   ...
#   QN (Noul, weight 1): actual_output completely
#                        follows this prompt instruction
#                        when responding to input.
#
# Each answer becomes a value in [0, 1] (the
# probability of "yes") and the score is their weighted
# mean. `metric.confidence` reports how decisive Jev
# was on average.
#
# NOTE: a system_one score comes from Jev's answers, not
# from the LLM-chain equation. Don't compare across
# eval modes.

prompt_instructions = [
    "Reply in all uppercase",
    "Mention the 30-day refund window",
]

system_one_prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=prompt_instructions,
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same clean run as before.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="WE OFFER A 30-DAY FULL REFUND AT NO EXTRA COST.",
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================

system_one_prompt_alignment.measure(test_case)
print("---")
print(f"Score:      {system_one_prompt_alignment.score}")
print(f"Confidence: {system_one_prompt_alignment.confidence}")
print(f"Reason:     {system_one_prompt_alignment.reason}")
print(f"Passed:     {system_one_prompt_alignment.is_successful()}")