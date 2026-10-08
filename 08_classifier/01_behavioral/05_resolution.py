from basemodel import CustomOpenAI
from deepeval.classifiers import ResolutionClassifier
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `ResolutionClassifier` is a built-in, categorical
# LLM-as-a-judge shipped by DeepEval for one specific
# question: by the end of the interaction, did the assistant
# bring the user's request to its expected end state, leave
# it unresolved, or hand it over? Its `name` and label set
# are fixed — that's the point of a built-in classifier, the
# labels mean the same thing across every dataset and run,
# so resolution rates are comparable between evaluations.
#
# The three labels split the space of conversation endings:
#
#   - "resolved"     — by the end of the interaction the
#                      user's goal is fully achieved
#                      (subscription cancelled, refund
#                      issued, answer found).
#   - "unresolved"   — by the end of the interaction the
#                      user's goal is not achieved and has
#                      not been handed over. The
#                      conversation just ends without a
#                      result.
#   - "handed_over"  — the interaction ends with the user
#                      handed over to a human or another
#                      channel instead of being resolved.
#
# The classifier is most at home on multi-turn test cases,
# where `scenario` (and optionally `expected_outcome`) tells
# the judge what a successful ending looks like, but it works
# on a single reply too. Per the default multi-turn template,
# the judge considers the conversation as a whole, not just
# the last turn — which is what makes "resolved" /
# "unresolved" / "handed_over" meaningful labels in the first
# place.
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

classifier = ResolutionClassifier(
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE CONVERSATIONAL TEST CASES
# ==========================================
# Unlike the other behavioral classifiers, this one takes
# `ConversationalTestCase` rather than `LLMTestCase`, because
# the label is a property of the *whole exchange*, not any
# individual reply. The judge reads every `Turn` along with
# `scenario` and `expected_outcome` to decide whether the
# user's goal was achieved.
#
# `expected_labels` is a dict keyed by `classifier.name`.
# For `ResolutionClassifier`, that key is fixed — the
# built-in `name` DeepEval assigns to this classifier — so
# you set the expected label as a string the judge will
# return.
#
# We exercise all three label values across three realistic
# end-to-end flows:
#
#   - "resolved"    — a subscription-cancellation flow
#                     where the assistant confirms, asks for
#                     one verification detail, and completes
#                     the cancellation. Goal fully achieved.
#   - "handed_over" — a refund flow where the assistant
#                     recognizes the case is outside its
#                     remit and transfers them to a human.
#                     Goal not achieved by the bot, but the
#                     handoff is the *correct* ending.
#   - "unresolved"  — a stuck-checkout flow where the
#                     assistant's suggestion doesn't work and
#                     the conversation simply ends without
#                     either resolution or escalation. This
#                     is the failure mode: the user is left
#                     hanging.
#
# `scenario` states the goal; `expected_outcome` (optional)
# spells out the substantive ending the metric is testing
# for.

test_case_resolved = ConversationalTestCase(
    scenario="User wants to cancel their subscription.",
    expected_outcome="The subscription is cancelled within the conversation.",
    turns=[
        Turn(role="user", content="I want to cancel my plan."),
        Turn(
            role="assistant",
            content=(
                "I can do that. Can you confirm the email on "
                "the account?"
            ),
        ),
        Turn(role="user", content="jane@example.com"),
        Turn(
            role="assistant",
            content=(
                "Done. Your plan is cancelled effective today."
            ),
        ),
    ],
    expected_labels={classifier.name: "resolved"},
)

test_case_handed_over = ConversationalTestCase(
    scenario="User wants a refund on a charge they don't recognise.",
    expected_outcome="The user is handed off to a human for the refund.",
    turns=[
        Turn(
            role="user",
            content=(
                "I was charged for a plan I cancelled three "
                "weeks ago. I want a refund."
            ),
        ),
        Turn(
            role="assistant",
            content=(
                "Refunds after cancellation need to be reviewed "
                "by our billing team. I'm connecting you now."
            ),
        ),
        Turn(role="user", content="Okay, thanks."),
    ],
    expected_labels={classifier.name: "handed_over"},
)

test_case_unresolved = ConversationalTestCase(
    scenario="User is stuck on a checkout error.",
    expected_outcome="The user gets a working checkout.",
    turns=[
        Turn(
            role="user",
            content="My checkout is stuck on 'processing'.",
        ),
        Turn(
            role="assistant",
            content=(
                "Can you try clearing your browser cache and "
                "retrying?"
            ),
        ),
        Turn(
            role="user",
            content="Tried, still stuck. Any other ideas?",
        ),
        Turn(role="assistant", content="Let me know if it works."),
    ],
    expected_labels={classifier.name: "unresolved"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# All three labels are exercised in one run, so the
# resolution rate, the handoff rate, and the unresolved
# rate come from the same judge and the same scale. That's
# the value of testing resolution through a fixed label set:
# drift in any one direction — the bot silently failing to
# close issues, or escalating cases it could have handled
# — becomes a measurable regression rather than a vibes-
# based review.

evaluate(
    test_cases=[
        test_case_resolved,
        test_case_handed_over,
        test_case_unresolved,
    ],
    classifiers=[classifier],
)