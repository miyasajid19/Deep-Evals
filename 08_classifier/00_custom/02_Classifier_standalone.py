from basemodel import CustomOpenAI
from deepeval.classifiers import Classifier, Label
from deepeval.test_case import LLMTestCase

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `Classifier.classify()` runs the judge once on a single
# test case and stores the result on the instance:
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
# that needs the label as a value.

classifier = Classifier(
    name="topic",
    model=CustomOpenAI(),
    include_reason=True,
    labels=[
        Label(
            name="billing",
            description="Questions about invoices, charges, or payment methods.",
        ),
        Label(
            name="refund",
            description="Requests to get money back for a purchase that has already been made.",
        ),
        Label(
            name="shipping",
            description="Questions about delivery status, times, or addresses.",
        ),
    ],
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# No `expected_labels` here — `classify()` runs the judge
# regardless. The label is recorded on the classifier
# instance for downstream code to read.

test_case = LLMTestCase(
    input="Where is my package? It was supposed to arrive yesterday.",
    actual_output="Your order is in transit and will arrive by Friday.",
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