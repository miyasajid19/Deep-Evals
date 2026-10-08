from basemodel import CustomOpenAI
from deepeval.classifiers import RequiredDisclosureClassifier
from deepeval.test_case import LLMTestCase

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `RequiredDisclosureClassifier.classify()` runs the judge
# once on a single test case and stores the result on
# the instance:
#   - classifier.label   — the label the judge picked
#                          ("present", "partial", or
#                          "missing"; or None if
#                          allow_none and the judge declined
#                          to classify).
#   - classifier.reason  — the judge's reasoning, or None
#                          when include_reason=False.
#   - classifier.error   — populated when the judge
#                          returned a label that is not one
#                          of the declared labels (an error,
#                          not a guess at the closest one).
#
# Use this when you want the classification itself, not a
# pass/fail verdict. `evaluate()` is for test reports; the
# standalone path is for routing, tagging, or any pipeline
# that needs to log / alert on a specific disclosure
# outcome (e.g. block a "missing" reply from going out the
# door in a regulated industry).

classifier = RequiredDisclosureClassifier(
    disclosures=[
        "a statement that this is not financial advice",
        "a recommendation to consult a licensed advisor",
    ],
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# No `expected_labels` here — `classify()` runs the judge
# regardless. The label is recorded on the classifier
# instance for downstream code to read.
#
# The test case mirrors the docs example: an investment
# question answered with a reply that includes both
# required disclosures. The expected behavior is
# "present" — both elements from the checklist appear in
# the reply.

test_case = LLMTestCase(
    input="Should I move my savings into index funds?",
    actual_output=(
        "Index funds are a common low-cost option for "
        "long-term savings. This isn't financial advice, "
        "so please talk to a licensed advisor about your "
        "situation before you decide."
    ),
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# `classify()` returns the chosen label and populates
# `classifier.label` / `classifier.reason`. We disable
# the progress indicator (`_show_indicator=False`) so
# the script output stays clean.

classifier.classify(test_case, _show_indicator=False)
print("---")
print(f"Label:   {classifier.label}")
print(f"Reason:  {classifier.reason}")
if classifier.error:
    print(f"Error:   {classifier.error}")