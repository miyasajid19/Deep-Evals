from basemodel import CustomOpenAI
from deepeval.classifiers import EscalationClassifier
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `EscalationClassifier` is a built-in, categorical
# LLM-as-a-judge shipped by DeepEval for one specific
# question: did the assistant hand the user to a human, offer
# to, or handle the request itself? Its `name` and label set
# are fixed — that's the point of a built-in classifier, the
# labels mean the same thing across every dataset and run, so
# escalation rates are comparable between evaluations.
#
# The three labels track the *progression* of escalation, not
# just whether it happened:
#
#   - "escalated"     — the response hands the user over to a
#                       human agent or another escalation path
#                       (warm transfer, ticket creation,
#                       "I'm connecting you now", ...).
#   - "offered"       — the response offers to escalate or
#                       connect the user with a human but does
#                       not do so yet ("would you like me to
#                       transfer you?", "I can loop in a
#                       specialist").
#   - "not_escalated" — the response handles the request
#                       itself with no mention of escalation.
#
# This progression matters because a support bot that always
# escalates is just a router, and one that never offers to is
# a liability when a real human is needed. Pairing each
# escalation trigger (angry customer, legal threat, out-of-
# scope request, mention of self-harm) with the behavior
# expected in that situation is what makes an escalation
# dataset work.
#
# Optional constructor knobs (defaults shown):
#   - model                  — evaluation LLM (defaults to
#                              DeepEval's default GPT model).
#                              Pass a `DeepEvalBaseLLM` to use
#                              your own model, e.g. the
#                              `CustomOpenAI` defined in
#                              `basemodel.py` for this repo.
#   - include_reason=True    — when True, the judge also
#                              returns a short reason explaining
#                              the chosen label.
#   - allow_none=False       — when False (the default), the
#                              judge always picks the closest
#                              label. Set True to let the judge
#                              abstain (`label=None` with a
#                              reason) when none fit.
#   - async_mode=True        — runs `classify()` concurrently
#                              across test cases.
#   - classification_template — override the judge prompt.
#   - eval_mode              — "llm" (default) or "system_one"
#                              (Jev-as-a-judge, requires
#                              `TYPESAFE_API_KEY`).

classifier = EscalationClassifier(
    model=CustomOpenAI(),
)
print(classifier.classification_template)
# ==========================================
# 2. THE TEST CASES
# ==========================================
# `expected_labels` is a dict keyed by `classifier.name`.
# For `EscalationClassifier`, that key is fixed — the
# built-in `name` DeepEval assigns to this classifier — so
# you set the expected label as a string the judge will
# return.
#
# We cover all three label values across the three most
# common escalation triggers:
#
#   - "escalated"        — a legal-threat input that the
#                          assistant actively hands off to a
#                          human ("I'm connecting you with a
#                          member of our team now"). The
#                          assistant has *done* the
#                          escalation, not merely offered.
#   - "offered"          — a billing-dispute input where the
#                          assistant offers to loop in a
#                          specialist but waits for the user
#                          to accept. The handoff has been
#                          proposed, not performed.
#   - "not_escalated"    — a routine order-status question
#                          that the assistant answers
#                          directly. No mention of escalation
#                          at all.
#
# `evaluate()` runs the judge on each case, prints a report,
# and writes it to the configured sink (Confident AI if
# connected, else the local cache). The case PASSES for
# this classifier when the returned label matches the
# expected one.

test_case_escalated = LLMTestCase(
    input="I'm going to sue you if this isn't fixed today.",
    actual_output=(
        "I understand. I'm connecting you with a member of "
        "our team now."
    ),
    expected_labels={classifier.name: "escalated"},
)

test_case_offered = LLMTestCase(
    input="I've been overcharged for three months in a row.",
    actual_output=(
        "I'm sorry to hear that. I can loop in our billing "
        "specialist to review your account — would you like "
        "me to do that?"
    ),
    expected_labels={classifier.name: "offered"},
)

test_case_not_escalated = LLMTestCase(
    input="Where is my package? It was supposed to arrive yesterday.",
    actual_output=(
        "Your order is in transit and will arrive by Friday."
    ),
    expected_labels={classifier.name: "not_escalated"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# All three labels are exercised in the same run, so the
# escalation rate, the offer rate, and the self-handle rate
# come from the same scale and the same judge. That's the
# value of testing escalation through a fixed label set:
# you can spot drift — say, the bot escalating everything
# or never offering when it should — without rebuilding
# the rubric each time.

evaluate(
    test_cases=[
        test_case_escalated,
        test_case_offered,
        test_case_not_escalated,
    ],
    classifiers=[classifier],
)