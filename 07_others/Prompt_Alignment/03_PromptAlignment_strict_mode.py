from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if EVERY instruction was followed
#   - 0 if any single instruction was missed
# It also overrides `threshold` to 1, so
# `is_successful()` matches the binary score exactly.
#
# Use it when any rule violation is unacceptable —
# compliance prompts, brand voice, regulatory
# disclaimers.

prompt_instructions = [
    "Reply in all uppercase",
    "Mention the 30-day refund window",
]

strict_prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=prompt_instructions,
    threshold=0.5,             # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. EVERY RULE FOLLOWED — score 1
# ==========================================
# Uppercase AND mentions the refund window. Expect 1.

perfect_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="WE OFFER A 30-DAY FULL REFUND AT NO EXTRA COST.",
)

strict_prompt_alignment.measure(perfect_case)
print("--- all rules followed ---")
print(f"Score:   {strict_prompt_alignment.score}")          # 1.0
print(f"Reason:  {strict_prompt_alignment.reason}")
print(f"Passed:  {strict_prompt_alignment.is_successful()}")  # True

# ==========================================
# 3. ONE RULE MISSED — score 0
# ==========================================
# Lowercase instead of uppercase. Strict mode drops
# the score to 0 even though the refund-window rule
# was followed.

lowercase_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="we offer a 30-day full refund at no extra cost.",
)

strict_prompt_alignment.measure(lowercase_case)
print("--- one rule missed ---")
print(f"Score:   {strict_prompt_alignment.score}")          # 0.0
print(f"Reason:  {strict_prompt_alignment.reason}")
print(f"Passed:  {strict_prompt_alignment.is_successful()}")  # False