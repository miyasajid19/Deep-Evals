from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the per-instruction
# classification (followed / not followed) before the
# final score. This is the fastest way to debug
# "the bot ignored rule #2" — verbose surfaces which
# rule was missed and the LLM's verdict.
#
# Keep `include_reason=True` (default) so the final
# reason lists the failed instructions too, in case
# verbose output scrolls off-screen.

prompt_instructions = [
    "Reply in all uppercase",
    "Mention the 30-day refund window",
    "Reply in under 30 words",
]

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=prompt_instructions,
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — partial compliance
# ==========================================
# The reply is uppercase and mentions the refund
# window, but it's 18 words — which is fine for the
# "< 30 words" rule, so all three rules pass. Expect
# score 1.0.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="WE OFFER A 30-DAY FULL REFUND AT NO EXTRA COST TO YOU.",
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason.

prompt_alignment.measure(test_case)
print("---")
print(f"Score:   {prompt_alignment.score}")
print(f"Reason:  {prompt_alignment.reason}")
print(f"Passed:  {prompt_alignment.is_successful()}")