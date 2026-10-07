from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval.metrics.faithfulness import FaithfulnessTemplate

# ==========================================
# 1. CUSTOM TEMPLATE
# ==========================================
# The default claim-extraction prompt asks the LLM to
# list every fact that "can be inferred" from the text.
# That's broad — it tends to extract implicit assumptions.
#
# Here we override `generate_claims` to only extract
# claims that are *asserted* (not inferred), with an
# example that returns an empty list when nothing is
# asserted. Subclass `FaithfulnessTemplate` and override
# the method whose prompt you want to replace. The other
# methods (`generate_truths`, `generate_verdicts`,
# `generate_reason`) keep the defaults.

class CustomFaithfulnessTemplate(FaithfulnessTemplate):
    @staticmethod
    def generate_claims(actual_output: str) -> str:
        return f"""Based on the given text, please extract a comprehensive list of facts that are explicitly ASSERTED (not inferred) by the provided text.

Example:
Example Text:
"CNN claims that the sun is 3 times smaller than earth."

Example JSON:
{{
    "claims": []
}}
===== END OF EXAMPLE ======

Text:
{actual_output}

JSON:
"""


# ==========================================
# 2. THE METRIC
# ==========================================
# Pass `evaluation_template=CustomFaithfulnessTemplate` so
# the metric uses the overridden `generate_claims` prompt.
# Everything else stays the same — truths extraction,
# verdicts, and reasoning still use the defaults.

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    evaluation_template=CustomFaithfulnessTemplate,
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# Two claims are asserted ("30-day refund", "initiate from
# order page") and both are supported by the retrieval
# context. With the assertion-only prompt, the LLM
# shouldn't pick up the implicit assumption about *how*
# refunds work.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "We offer a 30-day full refund at no extra cost. "
        "You can initiate a return from your order page. "
        "Refunds typically appear back on your card within 5-7 business days."
    ),
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
        "Returns can be initiated from the customer's order page in under a minute.",
        "Refunds are processed within 5-7 business days of the return arriving at our warehouse.",
    ],
)

# ==========================================
# 4. RUN STANDALONE
# ==========================================
faithfulness.measure(test_case)
print("---")
print(f"Score:   {faithfulness.score}")
print(f"Reason:  {faithfulness.reason}")
print(f"Passed:  {faithfulness.is_successful()}")