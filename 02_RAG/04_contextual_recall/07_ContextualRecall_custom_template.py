from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics.contextual_recall import ContextualRecallTemplate

# ==========================================
# 1. CUSTOM TEMPLATE
# ==========================================
# The default verdict prompt uses lenient phrasing — any
# sentence that's "supported by" or "derivable from" the
# context counts as attributable.
#
# Here we override `generate_verdicts` with a stricter
# rule: only sentences that are *directly stated* in
# `retrieval_context` count (no paraphrase, no inference).
# Note this metric's `generate_verdicts` takes only two
# args (`expected_output`, `retrieval_context`) — no
# `input`, unlike Contextual Precision. Subclass
# `ContextualRecallTemplate` and override the method
# whose prompt you want to replace. `generate_reason`
# keeps the default.

class CustomContextualRecallTemplate(ContextualRecallTemplate):
    @staticmethod
    def generate_verdicts(expected_output, retrieval_context):
        return f"""For EACH sentence in the given expected output below, determine whether the sentence is *directly stated* (not paraphrased, not inferred) by the nodes of retrieval contexts.

A sentence is directly stated only if its factual content is present verbatim or near-verbatim in some retrieval_context node.

Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "..."
        }},
    ]
}}

Expected Output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""


# ==========================================
# 2. THE METRIC
# ==========================================
# Pass `evaluation_template=CustomContextualRecallTemplate`
# so the metric uses the overridden `generate_verdicts`
# prompt. Everything else stays the same — the equation
# and `generate_reason` use the defaults.

contextual_recall = ContextualRecallMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    evaluation_template=CustomContextualRecallTemplate,
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# First expected sentence is directly stated; second
# is a paraphrase ("initiate from order page" vs. the
# context's "order history page"). With the strict
# template, only the first should be marked attributable.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output=(
        "You are eligible for a 30 day full refund at no extra cost. "
        "You can initiate a return from your order history page."
    ),
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Returns can be initiated from the customer's order page in under a minute.",
    ],
)

# ==========================================
# 4. RUN STANDALONE
# ==========================================
contextual_recall.measure(test_case)
print("---")
print(f"Score:   {contextual_recall.score}")
print(f"Reason:  {contextual_recall.reason}")
print(f"Passed:  {contextual_recall.is_successful()}")