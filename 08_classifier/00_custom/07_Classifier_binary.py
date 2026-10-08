from basemodel import CustomOpenAI
from deepeval.classifiers import Classifier
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. MULTI-LABEL AS BINARY CLASSIFIERS
# ==========================================
# A `Classifier` assigns exactly ONE label per test case.
# If your problem is genuinely multi-label — a single test
# case can be both `complaint` AND `question` — the
# recommended pattern is one classifier per label, each
# with a binary label set such as `["present", "absent"]`.
#
# Each classifier runs independently and its pass/fail is
# decided on its own `expected_labels` entry. Combining
# the per-label results lets you reconstruct a multi-label
# decision downstream.

complaint = Classifier(
    name="contains_complaint",
    model=CustomOpenAI(),
    include_reason=True,
    labels=["present", "absent"],
)

question = Classifier(
    name="contains_question",
    model=CustomOpenAI(),
    include_reason=True,
    labels=["present", "absent"],
)

escalation_request = Classifier(
    name="contains_escalation_request",
    model=CustomOpenAI(),
    include_reason=True,
    labels=["present", "absent"],
)

# ==========================================
# 2. THE TEST CASES
# ==========================================
# Three messages, each carrying zero, one, or more
# qualities. The expected label per classifier is the
# truth value for that base label on that message.
#
# `expected_labels` keys are the classifier names. The
# evaluate call below runs all three classifiers against
# all three messages and decides pass/fail per classifier
# per test case.

test_cases = [
    LLMTestCase(
        input="Thanks, the issue is resolved!",
        actual_output="Glad to hear it. Anything else I can help with?",
        expected_labels={
            complaint.name: "absent",
            question.name: "present",  # "Anything else I can help with?"
            escalation_request.name: "absent",
        },
    ),
    LLMTestCase(
        input="This is the third time I've reported this bug. Unacceptable.",
        actual_output="I understand your frustration. Let me look into this right away.",
        expected_labels={
            complaint.name: "present",
            question.name: "absent",
            escalation_request.name: "absent",
        },
    ),
    LLMTestCase(
        input="Can I please speak to a human? This bot isn't helping.",
        actual_output="Of course — transferring you to an agent now.",
        expected_labels={
            complaint.name: "present",  # "This bot isn't helping"
            question.name: "absent",
            escalation_request.name: "present",
        },
    ),
]

# ==========================================
# 3. RUN THE EVAL
# ==========================================
evaluate(
    test_cases=test_cases,
    classifiers=[complaint, question, escalation_request],
)