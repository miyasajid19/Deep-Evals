from basemodel import CustomOpenAI
from deepeval.classifiers import Classifier, Label
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# A `Classifier` also accepts a `ConversationalTestCase`.
# It reads every turn along with the scenario and expected
# outcome, and chooses ONE label for the conversation as a
# whole. The default multi-turn template explicitly tells
# the judge to consider the conversation as a whole, not
# just the last turn, unless a label description says
# otherwise.
#
# Use a multi-turn classifier when the outcome (topic,
# escalation category, sentiment at end of call) only makes
# sense after seeing how the conversation evolved. For
# per-turn classification, create multiple classifiers, one
# per turn position.

classifier = Classifier(
    name="escalation",
    model=CustomOpenAI(),
    include_reason=True,
    labels=[
        Label(
            name="self_service",
            description="The user's question was answered by the assistant without a human being brought in.",
        ),
        Label(
            name="escalated",
            description="A human agent was brought in, or the assistant explicitly handed off to one.",
        ),
        Label(
            name="unresolved",
            description="The conversation ended without the user's question being answered and without escalation.",
        ),
    ],
)

# ==========================================
# 2. THE CONVERSATIONAL TEST CASE
# ==========================================
# Three conversations, each labelled at the conversation
# level (not per turn):
#
#   1) self-service — the assistant answered the question
#   2) escalated   — the assistant handed off to a human
#   3) unresolved  — the user asked again and the assistant
#                    did not produce an answer
#
# `scenario` and `expected_outcome` are also read by the
# judge when present; populate them to give the judge more
# context about what the conversation was trying to do.

conversations = [
    ConversationalTestCase(
        scenario="A user asks about a recent charge on their credit card.",
        expected_outcome="The user learns which charge is which.",
        turns=[
            Turn(role="user", content="I see a $42 charge I don't recognise."),
            Turn(
                role="assistant",
                content="That was your Pro subscription renewal on the 12th. Anything else?",
            ),
            Turn(role="user", content="Ah, got it. Thanks!"),
        ],
        expected_labels={classifier.name: "self_service"},
    ),
    ConversationalTestCase(
        scenario="A user reports they cannot log in after a password reset.",
        expected_outcome="A human agent takes over.",
        turns=[
            Turn(role="user", content="I reset my password but still can't log in."),
            Turn(
                role="assistant",
                content="Let me transfer you to a human agent who can verify your account.",
            ),
            Turn(role="user", content="Okay, thanks."),
        ],
        expected_labels={classifier.name: "escalated"},
    ),
    ConversationalTestCase(
        scenario="A user is stuck on a checkout error.",
        expected_outcome="The user gets a working answer.",
        turns=[
            Turn(role="user", content="My checkout is stuck on 'processing'."),
            Turn(
                role="assistant",
                content="Can you try clearing your browser cache and retrying?",
            ),
            Turn(role="user", content="Tried, still stuck. Any other ideas?"),
            Turn(role="assistant", content="Let me know if it works."),
        ],
        expected_labels={classifier.name: "unresolved"},
    ),
]

# ==========================================
# 3. RUN THE EVAL
# ==========================================
evaluate(test_cases=conversations, classifiers=[classifier])