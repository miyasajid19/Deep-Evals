from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import PromptAlignmentMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# PromptAlignmentMetric checks whether your LLM's
# `actual_output` follows each rule in
# `prompt_instructions` — the literal instructions you
# embedded in your prompt template.
#
# Algorithm (default `llm` eval mode):
#   1) for each instruction, the LLM classifies whether
#      the actual_output follows it (yes/no), with
#      context from the input.
#   2) score = number of instructions followed /
#             total number of instructions
#
# `prompt_instructions` is MANDATORY — that's the list
# of rules you want obeyed. Each item is a single
# atomic rule; bundling several into one item forces an
# all-or-nothing verdict.
#
# The metric is `referenceless`: no `expected_output`
# needed.

# ==========================================
# 2. THE PROMPT INSTRUCTIONS
# ==========================================
# One atomic rule per entry. "Reply in all uppercase"
# is a clean check the LLM can clearly follow or
# ignore.

prompt_instructions = [
    "Reply in all uppercase",
]

prompt_alignment = PromptAlignmentMetric(
    prompt_instructions=prompt_instructions,  # ← mandatory
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# The reply is in all caps, so the rule is followed.
# Expect score 1.0.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="WE OFFER A 30-DAY FULL REFUND AT NO EXTRA COST.",
)

# ==========================================
# 4. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and
# writes a report (Confident AI if configured, else
# local cache).

evaluate(test_cases=[test_case], metrics=[prompt_alignment])