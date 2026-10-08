from basemodel import CustomOpenAI
from deepeval.classifiers import InstructionCompletenessClassifier
from deepeval.test_case import LLMTestCase

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# You can also run the `InstructionCompletenessClassifier`
# on a single test case as a one-off execution. `classify()`
# returns the label and stores the result on the instance:
#   - classifier.label   — the label the judge picked
#                          ("complete", "partial", or
#                          "ignored"; or None if
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
# that needs the label as a value (e.g. routing partial
# completions back to the assistant for a second pass
# before they reach the user).

classifier = InstructionCompletenessClassifier(
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
# the assistant to summarize the meeting notes, list the
# action items, AND suggest a date for the follow-up.
# The expected judge output is "complete" — the response
# addresses all three parts.

test_case = LLMTestCase(
    input=(
        "Summarize the meeting notes, list the action "
        "items, and suggest a date for the follow-up."
    ),
    actual_output=(
        "Summary: the team agreed to revisit pricing and "
        "the launch timeline. Action items: 1. Priya to "
        "draft the new pricing tiers. 2. Marco to share "
        "the revised launch timeline. 3. Lee to book the "
        "follow-up. Suggested follow-up: next Tuesday."
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