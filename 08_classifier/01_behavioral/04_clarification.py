from basemodel import CustomOpenAI
from deepeval.classifiers import ClarificationClassifier
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `ClarificationClassifier` is a built-in, categorical
# LLM-as-a-judge shipped by DeepEval for one specific
# question: did the assistant pause to ask a clarifying
# question on an ambiguous input, answer because the input
# was clear, or barrel ahead on an assumption? Its `name` and
# label set are fixed — that's the point of a built-in
# classifier, the labels mean the same thing across every
# dataset and run, so clarification rates are comparable
# between evaluations.
#
# The three labels split the joint of input ambiguity and
# response behavior:
#
#   - "asked_clarification" — the input is ambiguous or
#                             underspecified and the response
#                             asks a clarifying question
#                             before proceeding.
#   - "answered_directly"   — the input is clear enough and
#                             the response answers or acts on
#                             it without needing
#                             clarification.
#   - "guessed"             — the input is ambiguous or
#                             underspecified but the response
#                             proceeds on an assumption
#                             instead of asking. This is the
#                             costly failure mode for agents
#                             that take actions (bookings,
#                             purchases, data changes) —
#                             acting on the wrong assumption
#                             is what makes guessing
#                             dangerous, not the guessing
#                             itself.
#
# Underspecified prompts are the right test cases here,
# because they show whether the assistant pauses. A clear
# prompt doesn't test clarification behavior at all — the
# right answer is always to answer.
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

classifier = ClarificationClassifier(
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASES
# ==========================================
# `expected_labels` is a dict keyed by `classifier.name`.
# For `ClarificationClassifier`, that key is fixed — the
# built-in `name` DeepEval assigns to this classifier — so
# you set the expected label as a string the judge will
# return.
#
# We exercise all three label values, and crucially the
# first two test cases use the *same* underspecified
# "Book me a flight to Portland next week" prompt so the
# judge has to lean on the *response* to pick the label,
# not the prompt:
#
#   - "asked_clarification" — the same ambiguous flight
#                             request is answered by asking
#                             which Portland and which day.
#                             The pause worked.
#   - "guessed"             — the same ambiguous flight
#                             request is answered by booking
#                             a specific flight. This is the
#                             failure mode the classifier
#                             exists to catch: the prompt
#                             hadn't changed, the response
#                             did.
#   - "answered_directly"   — a fully specified prompt
#                             ("Portland, Maine, Friday,
#                             morning flight, one passenger")
#                             is answered with a booking.
#                             The prompt was clear, so
#                             answering directly was correct;
#                             asking again here would itself
#                             be a failure.
#
# `evaluate()` runs the judge on each case, prints a report,
# and writes it to the configured sink (Confident AI if
# connected, else the local cache). The case PASSES for
# this classifier when the returned label matches the
# expected one.

test_case_asked_clarification = LLMTestCase(
    input="Book me a flight to Portland next week.",
    actual_output=(
        "Portland, Oregon or Portland, Maine? And which day "
        "next week works best?"
    ),
    expected_labels={classifier.name: "asked_clarification"},
)

test_case_guessed = LLMTestCase(
    input="Book me a flight to Portland next week.",
    actual_output=(
        "Booked — Delta 1841, Portland (PDX), departing "
        "Monday at 06:30. Confirmation ABC123."
    ),
    expected_labels={classifier.name: "guessed"},
)

test_case_answered_directly = LLMTestCase(
    input=(
        "Book me a flight from Boston to Portland, Maine, "
        "next Friday morning, one passenger."
    ),
    actual_output=(
        "Booked — JetBlue 482, Portland (PWM), departing "
        "Friday at 07:55. Confirmation ABC234."
    ),
    expected_labels={classifier.name: "answered_directly"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# All three labels are exercised in one run. The first two
# cases share a prompt on purpose: when the input is
# ambiguous, the judge has to score the response, not the
# prompt. That's the only way the "asked_clarification" vs.
# "guessed" distinction is measurable — if the prompts
# differed, the judge could just lean on prompt clarity and
# the metric would degrade to a prompt-clarity detector.

evaluate(
    test_cases=[
        test_case_asked_clarification,
        test_case_guessed,
        test_case_answered_directly,
    ],
    classifiers=[classifier],
)