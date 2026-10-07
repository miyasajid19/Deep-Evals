from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate

# ==========================================
# 1. CUSTOM TEMPLATE
# ==========================================
# The default extraction prompt splits `actual_output` into
# atomic statements, then the LLM classifies each one. We
# override `generate_statements` to add a domain-specific
# instruction: in this RAG bot's context, treat each bullet
# in a bulleted list as its own statement (the default
# sometimes merges adjacent bullets).
#
# Subclass `AnswerRelevancyTemplate` and override the
# method whose prompt you want to replace. The other
# methods (`generate_verdicts`, `generate_reason`) keep
# the defaults.

class CustomAnswerRelevancyTemplate(AnswerRelevancyTemplate):
    @staticmethod
    def generate_statements(actual_output: str) -> str:
        return f"""Given the text, extract every atomic statement presented.
Treat each bullet point in a bulleted list as its own statement.
Treat each sentence in a paragraph as its own statement.

Example:
Our new laptop model features a high-resolution Retina display for crystal-clear visuals.
It also has 16GB of RAM and an M2 processor.

{{
    "statements": [
        "The new laptop model has a high-resolution Retina display.",
        "It produces crystal-clear visuals.",
        "The new laptop model has 16GB of RAM.",
        "The new laptop model has an M2 processor."
    ]
}}
===== END OF EXAMPLE ======

Text:
{actual_output}

JSON:
"""


# ==========================================
# 2. THE METRIC
# ==========================================
# Pass `evaluation_template=CustomAnswerRelevancyTemplate`
# so the metric uses the overridden `generate_statements`
# prompt. Everything else stays the same — verdicts and
# reasoning still use the defaults.

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    evaluation_template=CustomAnswerRelevancyTemplate,
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# Bulleted answer — the custom template splits each bullet
# into its own statement, so a bullet that's off-topic
# will drop the score instead of being merged with a
# relevant neighbor.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "Here's what we offer:\n"
        "- 30-day full refund at no extra cost.\n"
        "- Free return shipping within the US.\n"
        "- A loyalty program with 5% cashback on future orders."
    ),
)

# ==========================================
# 4. RUN STANDALONE
# ==========================================
answer_relevancy.measure(test_case)
print("---")
print(f"Score:   {answer_relevancy.score}")
print(f"Reason:  {answer_relevancy.reason}")
print(f"Passed:  {answer_relevancy.is_successful()}")