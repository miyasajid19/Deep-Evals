from basemodel import CustomOpenAI
from deepeval.classifiers import ResponseLanguageClassifier
from deepeval.test_case import LLMTestCase

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# You can also run the `ResponseLanguageClassifier` on a
# single test case as a one-off execution. `classify()`
# returns the label and stores the result on the instance:
#   - classifier.label   — the label the judge picked
#                          ("matches_user" or "mismatch";
#                          or None if allow_none and the
#                          judge declined to classify).
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
# that needs the label as a value (e.g. flagging every
# response that drifted to English for a translation
# review before it reaches the user).

classifier = ResponseLanguageClassifier(
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# No `expected_labels` here — `classify()` runs the judge
# regardless. The label is recorded on the classifier
# instance for downstream code to read.
#
# The test case mirrors the docs example: the user asks
# for the order status in Spanish, and the assistant
# replies in Spanish. The expected judge output is
# "matches_user" — the response language matches the
# input language.

test_case = LLMTestCase(
    input="¿Dónde está mi pedido?",
    actual_output=(
        "Su pedido fue enviado ayer y llegará el jueves."
    ),
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