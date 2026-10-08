from basemodel import CustomOpenAI
from deepeval.classifiers import InstructionCompletenessClassifier
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `InstructionCompletenessClassifier` is a built-in,
# categorical LLM-as-a-judge shipped by DeepEval for one
# specific question: when the user packs several explicit
# parts into one request, did the assistant address every
# part, only some of them, or none? Its `name` and label
# set are fixed — that's the point of a built-in
# classifier, the labels mean the same thing across every
# dataset and every run, so completeness rates are
# comparable between evaluations.
#
# The three labels cover the full spectrum from "got
# everything" to "got nothing":
#
#   - "complete" — every part of the user's request is
#                  addressed in the response. The right
#                  answer.
#   - "partial"  — some parts of the user's request are
#                  addressed and others are not. The
#                  dangerous middle ground — a model that
#                  answers the first sub-task and silently
#                  drops the rest is more dangerous than
#                  one that drops everything, because the
#                  user only notices the missing piece
#                  after relying on it.
#   - "ignored"  — none of the parts of the user's request
#                  are addressed. The response went off
#                  in a different direction entirely.
#
# The judge inspects only `input` (the user's multi-part
# request) and `actual_output` (the assistant's reply).
# The classifier is most useful for compound tasks such
# as "summarize this, list the action items, then draft a
# reply" — that is where an assistant tends to handle the
# first sub-task and quietly drop the others.
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

classifier = InstructionCompletenessClassifier(
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASES
# ==========================================
# `expected_labels` is a dict keyed by `classifier.name`.
# For `InstructionCompletenessClassifier`, that key is
# fixed — the built-in `name` DeepEval assigns to this
# classifier — so you set the expected label as a string
# the judge will return.
#
# All three cases share the same three-part user request
# ("summarize, list, suggest a date") so the judge scores
# the *response* — the only thing that varies between
# cases is what the assistant did with the three parts.
# That is the only way the distinction between complete,
# partial, and ignored is measurable.
#
#   - "complete" — the assistant returns a summary, the
#                  action items, AND a suggested follow-up
#                  date. All three parts are addressed.
#   - "partial"  — the assistant returns a summary and
#                  the action items, but no suggested
#                  date. The third part of the request
#                  was silently dropped — exactly the
#                  failure mode this label catches.
#   - "ignored"  — the assistant returns something
#                  unrelated (a recipe for chocolate cake)
#                  and addresses none of the three parts.
#                  The request went in one ear and out the
#                  other.
#
# `evaluate()` runs the judge on each case, prints a
# report, and writes it to the configured sink (Confident
# AI if connected, else the local cache). The case PASSES
# for this classifier when the returned label matches the
# expected one.

test_case_complete = LLMTestCase(
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
    expected_labels={classifier.name: "complete"},
)

test_case_partial = LLMTestCase(
    input=(
        "Summarize the meeting notes, list the action "
        "items, and suggest a date for the follow-up."
    ),
    actual_output=(
        "Summary: the team agreed to revisit pricing and "
        "the launch timeline. Action items: 1. Priya to "
        "draft the new pricing tiers. 2. Marco to share "
        "the revised launch timeline."
    ),
    expected_labels={classifier.name: "partial"},
)

test_case_ignored = LLMTestCase(
    input=(
        "Summarize the meeting notes, list the action "
        "items, and suggest a date for the follow-up."
    ),
    actual_output=(
        "Here's a recipe for chocolate cake: whisk "
        "together flour, sugar, cocoa, eggs, and butter, "
        "then bake at 180C for 35 minutes."
    ),
    expected_labels={classifier.name: "ignored"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# All three labels are exercised in the same run, so the
# complete rate, the partial rate, and the ignore rate
# come from the same judge and the same scale. That's the
# value of testing instruction completeness through a
# fixed label set: drift — say, the model silently
# dropping the last sub-task of every compound request —
# becomes a measurable regression rather than a
# vibes-based review.

evaluate(
    test_cases=[
        test_case_complete,
        test_case_partial,
        test_case_ignored,
    ],
    classifiers=[classifier],
)