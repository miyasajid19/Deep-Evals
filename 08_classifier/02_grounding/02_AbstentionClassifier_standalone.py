from basemodel import CustomOpenAI
from deepeval.classifiers import AbstentionClassifier
from deepeval.test_case import LLMTestCase

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# You can also run the `AbstentionClassifier` on a single
# test case as a one-off execution. `classify()` returns
# the label and stores the result on the instance:
#   - classifier.label   — the label the judge picked
#                          (a string, or None if allow_none
#                          and the judge declined to classify)
#   - classifier.reason  — the judge's reasoning, or None
#                          when include_reason=False
#   - classifier.error   — populated when the judge returned
#                          a label that is not one of the
#                          declared labels (an error, not a
#                          guess at the closest one)
#
# Use this when you want the classification itself, not a
# pass/fail verdict. `evaluate()` is for test reports; the
# standalone path is for routing, tagging, or any pipeline
# that needs the label as a value (e.g. flagging fabricated
# answers for review before they ship).

classifier = AbstentionClassifier(
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# `retrieval_context` is populated so the judge can see
# what the application actually had to work with. Without
# context the classifier cannot separate "abstained" from
# "fabricated" — it has no way to know the answer was
# missing.
#
# The assistant here was asked about enterprise refund
# windows; the context only covers standard plans, and the
# response invents a 90-day enterprise window. That makes
# this a "fabricated" case: the context did not have it,
# but the response made one up.

test_case = LLMTestCase(
    input="What is the refund window for enterprise plans?",
    actual_output=(
        "Enterprise plans can be refunded within 90 days "
        "of purchase — just file a ticket and we'll "
        "process it."
    ),
    retrieval_context=[
        "Standard plans can be refunded within 30 days of "
        "purchase. Enterprise plans are handled by the "
        "account team and are not covered by this policy."
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# `classify()` returns the chosen label and populates
# `classifier.label` / `classifier.reason`. We disable
# the progress indicator (`_show_indicator=False`) so the
# script output stays clean.

classifier.classify(test_case, _show_indicator=False)
print("---")
print(f"Label:   {classifier.label}")
print(f"Reason:  {classifier.reason}")
if classifier.error:
    print(f"Error:   {classifier.error}")