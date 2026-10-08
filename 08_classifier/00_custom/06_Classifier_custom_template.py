from basemodel import CustomOpenAI
from deepeval.classifiers import Classifier, ClassifierTemplate
from deepeval.test_case import LLMTestCase
import textwrap

# ==========================================
# 1. CUSTOM TEMPLATE
# ==========================================
# The default prompts (under
# `deepeval/classifiers/classifier/templates/`) ask the
# judge to pick ONE label and emit JSON with `label` and
# `reason`. They are a sensible default, but you can
# override them by subclassing `ClassifierTemplate`.
#
# Two methods are available — one per test case type:
#   - `classify_single_turn(labels, test_case_content)`
#   - `classify_multi_turn(labels, test_case_content, turns)`
#
# `labels` is the rendered list of label names and
# descriptions, `test_case_content` is the rendered test
# case fields, and `turns` is a list of dicts with the
# populated fields of each turn. An override only needs to
# declare the variables it uses, and must keep asking for
# JSON with `label` and `reason` fields — matching is exact
# (case-insensitive) against your declared labels, and
# anything else is an error, not a guess.
#
# In this example we add a tie-breaker rule: when two
# labels are equally applicable, prefer the more specific
# one. This is the kind of guidance the default template
# leaves implicit.

class StrictClassifierTemplate(ClassifierTemplate):
    @staticmethod
    def classify_single_turn(labels: str, test_case_content: str) -> str:
        return textwrap.dedent(
            f"""
            Pick exactly one label for the test case below.
            Tie-breaker rule: when two labels seem equally
            applicable, prefer the more specific one.

            Labels:
            {labels}

            Test case:
            {test_case_content}

            Return JSON only with two fields:
              - "label":   one of the label names above
              - "reason":  a brief explanation for the choice
            JSON:
            """
        )

    @staticmethod
    def classify_multi_turn(
        labels: str, test_case_content: str, turns: list
    ) -> str:
        return textwrap.dedent(
            f"""
            Pick exactly one label for the conversation
            below. Consider the conversation as a whole, not
            just the last turn. When two labels seem equally
            applicable, prefer the more specific one.

            Labels:
            {labels}

            {test_case_content if test_case_content else ""}
            Turns:
            {turns}

            Return JSON only with two fields:
              - "label":   one of the label names above
              - "reason":  a brief explanation for the choice
            JSON:
            """
        )


# ==========================================
# 2. THE CLASSIFIER
# ==========================================
# Pass `classification_template=StrictClassifierTemplate`
# so the metric uses the overridden prompts. Everything
# else stays the same — labels, model, eval mode.

classifier = Classifier(
    name="topic",
    model=CustomOpenAI(),
    include_reason=True,
    labels=["billing", "refund", "shipping"],
    classification_template=StrictClassifierTemplate,
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# A test case that lives on the edge between `billing`
# (questions about invoices / charges) and `refund`
# (requests to get money back). The custom template tells
# the judge to prefer the more specific one — that's
# `refund` here, because the user is asking for their
# money back rather than asking about a charge.

test_case = LLMTestCase(
    input="Can I get my money back for last month's subscription?",
    actual_output="Yes — I can refund the last charge on your account.",
    expected_labels={classifier.name: "refund"},
)

# ==========================================
# 4. RUN STANDALONE
# ==========================================
classifier.classify(test_case, _show_indicator=False)
print("---")
print(f"Label:   {classifier.label}")
print(f"Reason:  {classifier.reason}")
if classifier.error:
    print(f"Error:   {classifier.error}")