from basemodel import CustomOpenAI
from deepeval.classifiers import Classifier
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# Plain strings are fine when the label name is unambiguous
# on its own — there's no boundary between labels to spell
# out. `Classifier` accepts `Sequence[Union[str, Label]]`
# and converts strings to `Label(name=...)` with no
# description.
#
# Sentiment is a good fit for strings: positive / neutral /
# negative are mutually exclusive and the names carry the
# meaning. Compare this with the topic classifier in
# `01_Classifier_basic.py`, where the descriptions do the
# boundary work that names alone can't do ("billing" vs
# "refund" is not obvious from the names).

classifier = Classifier(
    name="sentiment",
    model=CustomOpenAI(),
    include_reason=True,
    labels=["positive", "neutral", "negative"],
)

# ==========================================
# 2. THE TEST CASES
# ==========================================
# Three customer reviews spanning the three labels. The
# third one is a mild complaint — the judge should still
# classify it as negative rather than neutral because the
# user is dissatisfied.

test_cases = [
    LLMTestCase(
        input="This app changed my life. Five stars.",
        actual_output="Thanks for the kind words!",
        expected_labels={classifier.name: "positive"},
    ),
    LLMTestCase(
        input="What are the supported export formats?",
        actual_output="The app supports CSV, JSON, and PDF export.",
        expected_labels={classifier.name: "neutral"},
    ),
    LLMTestCase(
        input="The latest update broke my workflow. I want a refund.",
        actual_output="I'm sorry to hear that. Let me help you get this sorted.",
        expected_labels={classifier.name: "negative"},
    ),
]

# ==========================================
# 3. RUN THE EVAL
# ==========================================
evaluate(test_cases=test_cases, classifiers=[classifier])