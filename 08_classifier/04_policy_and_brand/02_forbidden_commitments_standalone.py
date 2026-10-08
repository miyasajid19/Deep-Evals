from basemodel import CustomOpenAI
from deepeval.classifiers import ForbiddenCommitmentsClassifier
from deepeval.test_case import LLMTestCase

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `ForbiddenCommitmentsClassifier.classify()` runs the
# judge once on a single test case and stores the result
# on the instance:
#   - classifier.label   — the label the judge picked
#                          ("clean", "unauthorized_commitment",
#                          "competitor_mention",
#                          "disparagement"; or None if
#                          allow_none and the judge declined
#                          to classify).
#   - classifier.reason  — the judge's reasoning, or None
#                          when include_reason=False.
#   - classifier.error   — populated when the judge returned
#                          a label that is not one of the
#                          declared labels (an error, not a
#                          guess at the closest one).
#
# Use this when you want the classification itself, not a
# pass/fail verdict. `evaluate()` is for test reports; the
# standalone path is for routing, tagging, or any pipeline
# that needs to log / alert on a specific policy outcome
# (e.g. auto-flag a refund-promising reply for human review
# before it goes out the door).

classifier = ForbiddenCommitmentsClassifier(
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# No `expected_labels` here — `classify()` runs the judge
# regardless. The label is recorded on the classifier
# instance for downstream code to read.
#
# The test case mirrors the docs example: a user asks for
# an immediate refund. The expected behavior is "clean" —
# the assistant should deflect to the billing team without
# committing to the refund itself.

test_case = LLMTestCase(
    input="Can you just give me a full refund right now?",
    actual_output=(
        "I can't approve refunds myself, but I've opened "
        "a request and the billing team will review it "
        "within two business days."
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