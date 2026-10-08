from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Prompt Alignment: the LLM extraction step
# is preserved (parsing the actual_output for each
# instruction's verdict context). The actual
# classification — "did the actual_output follow this
# instruction?" — is delegated to Jev, one yes/no
# question per instruction, with `P(yes) >= 0.5`
# counted as followed.
#
# The equation is unchanged:
#   Score = N followed / N total
# And the LLM-written reason is unchanged. If a Jev
# call fails, the metric silently defers that
# classification to the LLM.

prompt_instructions = [
    "Reply in all uppercase",
    "Mention the 30-day refund window",
]

hybrid_prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=prompt_instructions,
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still extracts verdict context
    include_reason=True,
    eval_mode="hybrid",         # Jev classifies each instruction
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same clean run as the basic example — uppercase and
# mentions the refund window. Expect score 1.0.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="WE OFFER A 30-DAY FULL REFUND AT NO EXTRA COST.",
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# `metric.confidence` reports how decisive Jev was on
# average across its per-instruction calls.

hybrid_prompt_alignment.measure(test_case)
print("---")
print(f"Score:      {hybrid_prompt_alignment.score}")
print(f"Confidence: {hybrid_prompt_alignment.confidence}")
print(f"Reason:     {hybrid_prompt_alignment.reason}")
print(f"Passed:     {hybrid_prompt_alignment.is_successful()}")