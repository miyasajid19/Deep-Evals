from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics.contextual_precision import ContextualPrecisionTemplate

# ==========================================
# 1. CUSTOM TEMPLATE
# ==========================================
# Contextual Precision's only LLM step is the per-node
# verdict — "was this node useful for arriving at the
# expected output?". The default prompt uses the word
# "remotely" so any tangential node counts as useful.
#
# Here we override `generate_verdicts` with a stricter
# criterion: a node only counts as useful if the
# expected output cites it (i.e. its content is
# *necessary*, not just adjacent). Subclass
# `ContextualPrecisionTemplate` and override the method
# whose prompt you want to replace. `generate_reason`
# keeps the default.

class CustomContextualPrecisionTemplate(ContextualPrecisionTemplate):
    @staticmethod
    def generate_verdicts(input, expected_output, retrieval_context):
        return f"""Given the input, expected output, and retrieval context, please generate a list of JSON objects to determine whether each node in the retrieval context is *necessary* for arriving at the expected output.

A node is necessary only if a fact it contains is cited or directly relied on in the expected output. Tangentially related nodes do NOT count.

Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "..."
        }}
    ]
}}
The number of 'verdicts' SHOULD BE STRICTLY EQUAL to that of the contexts.
**

Input:
{input}

Expected output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""


# ==========================================
# 2. THE METRIC
# ==========================================
# Pass `evaluation_template=CustomContextualPrecisionTemplate`
# so the metric uses the overridden `generate_verdicts`
# prompt. Everything else stays the same — the WCP equation
# and `generate_reason` use the defaults.

contextual_precision = ContextualPrecisionMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    evaluation_template=CustomContextualPrecisionTemplate,
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# Two retrieved nodes. The first one is necessary for the
# expected answer; the second is adjacent but irrelevant
# to the actual refund question. With the strict
# template, only the first should be marked useful.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",  # necessary
        "Our loyalty program gives 5% cashback on future orders.",                 # adjacent but not cited
    ],
)

# ==========================================
# 4. RUN STANDALONE
# ==========================================
contextual_precision.measure(test_case)
print("---")
print(f"Score:   {contextual_precision.score}")
print(f"Reason:  {contextual_precision.reason}")
print(f"Passed:  {contextual_precision.is_successful()}")