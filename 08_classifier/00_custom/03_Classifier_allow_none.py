from basemodel import CustomOpenAI
from deepeval.classifiers import Classifier, Label
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# By default the judge must pick one of your labels even
# when the fit is imperfect. When `allow_none=True`, the
# judge is allowed to return `label=None` (the sentinel
# label "NONE") when none of the labels apply.
#
# Use this instead of adding an "other" bucket. An "other"
# label invites the judge to use it as a catch-all, which
# makes your label set less informative. With
# `allow_none=True`:
#   - `classifier.label` is None when the judge declines
#   - the test case fails only when you expected a
#     specific label and the judge returned None
#   - leaving `expected_labels` empty on a test case
#     records the classification without passing/failing

classifier = Classifier(
    name="topic",
    model=CustomOpenAI(),
    include_reason=True,
    allow_none=True,
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
# 2. THE TEST CASES
# ==========================================
# Three cases that demonstrate the behavior:
#   1) on-topic shipping question → label "shipping"
#   2) on-topic billing question  → label "billing"
#   3) off-topic greeting that fits none of the labels
#      → label None (the "NONE" sentinel)
#
# For the off-topic case we OMIT the expected label rather
# than setting it to "NONE": deepeval validates
# `expected_labels` against the classifier's declared
# label set before the judge runs, so an unknown value
# like "NONE" raises immediately. Leaving
# `expected_labels` empty records the classification but
# does not pass/fail the case — which is exactly what you
# want when the only thing under test is "the judge
# returned no label".

on_topic_shipping = LLMTestCase(
    input="Where's my package?",
    actual_output="It's out for delivery and will arrive by 5pm today.",
    expected_labels={classifier.name: "shipping"},
)

on_topic_billing = LLMTestCase(
    input="Can I change the card on file?",
    actual_output="Yes — go to Settings > Payment to update it.",
    expected_labels={classifier.name: "billing"},
)

off_topic = LLMTestCase(
    input="Hello!",
    actual_output="Hi there! How can I help?",
    # No expected label — the judge returning None is
    # recorded on the run, but the case does not pass/fail.
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
evaluate(
    test_cases=[on_topic_shipping, on_topic_billing, off_topic],
    classifiers=[classifier],
)