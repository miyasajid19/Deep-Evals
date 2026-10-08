from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single metric instance scoring three responses at
# varying levels of rule compliance. Each response
# claims to handle the same input but follows a
# different share of the instructions.
#
# Score = N followed / N total. There is no weighting —
# every instruction in `prompt_instructions` is a
# pass/fail binary.

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
)

# ==========================================
# 2. CASE A — all three rules followed
# ==========================================
# Uppercase, mentions refund, 8 words. Expect 3/3 = 1.0.

all_followed = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="WE OFFER A 30-DAY FULL REFUND AT NO EXTRA COST TO YOU.",
)

# ==========================================
# 3. CASE B — two of three followed
# ==========================================
# Uppercase, mentions refund, but 36 words (over the
# 30-word limit). Expect 2/3 ≈ 0.667.

two_followed = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "WE OFFER A 30-DAY FULL REFUND AT NO EXTRA COST TO YOU, "
        "AND CUSTOMER SERVICE IS AVAILABLE TWENTY-FOUR HOURS A DAY "
        "FOR ALL YOUR QUESTIONS AND CONCERNS."
    ),
)

# ==========================================
# 4. CASE C — none of the three followed
# ==========================================
# Lowercase, no refund mention, 36 words. Expect 0/3 = 0.

none_followed = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "we offer customer support available twenty-four hours a day "
        "for all your questions and concerns, with no extra cost to "
        "you during business hours and beyond normal business hours."
    ),
)

# ==========================================
# 5. RUN THE METRIC ON ALL THREE CASES
# ==========================================

def run(label, case):
    prompt_alignment.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {prompt_alignment.score:.3f}")
    print(f"  Passed: {prompt_alignment.is_successful()}")
    print(f"  Reason: {prompt_alignment.reason}")

run("A — all three rules followed", all_followed)
run("B — two of three followed",   two_followed)
run("C — none of the three",       none_followed)