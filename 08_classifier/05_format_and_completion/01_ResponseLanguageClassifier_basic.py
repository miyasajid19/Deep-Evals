from basemodel import CustomOpenAI
from deepeval.classifiers import ResponseLanguageClassifier
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `ResponseLanguageClassifier` is a built-in, categorical
# LLM-as-a-judge shipped by DeepEval for one specific
# question: when the user writes in language X, does the
# assistant reply in language X? Its `name` and label set
# are fixed — that's the point of a built-in classifier,
# the labels mean the same thing across every dataset and
# every run, so language-match rates are comparable
# between evaluations.
#
# The two labels cover both directions of the boundary:
#
#   - "matches_user" — the response is written in the
#                      same language as the user's input.
#                      The right answer.
#   - "mismatch"     — the response is written in a
#                      different language from the user's
#                      input. The failure mode the
#                      classifier exists to catch: a model
#                      drifting back to English (or to
#                      whatever language its system prompt
#                      was written in) when the user wrote
#                      in something else.
#
# The judge inspects only `input` (the user's text) and
# `actual_output` (the assistant's reply) — no context,
# no retrieval, no tools. That is what makes this
# classifier useful for multilingual user-facing apps:
# it scores the surface form of the conversation, not
# the reasoning behind it.
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

classifier = ResponseLanguageClassifier(
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASES
# ==========================================
# `expected_labels` is a dict keyed by `classifier.name`.
# For `ResponseLanguageClassifier`, that key is fixed —
# the built-in `name` DeepEval assigns to this classifier
# — so you set the expected label as a string the judge
# will return.
#
# We exercise both label values in one run. Both pairs
# pick languages the judge model can be expected to
# recognise reliably (Spanish, French) so the demo
# isolates the classifier, not the model.
#
#   - "matches_user" — the user asks in Spanish, the
#                      assistant replies in Spanish. The
#                      response language matches the input
#                      language, which is the behavior
#                      the classifier is meant to confirm.
#   - "mismatch"     — the user asks in Spanish, the
#                      assistant replies in English. The
#                      response language does not match the
#                      input language, which is the
#                      failure mode the classifier is
#                      meant to catch.
#
# `evaluate()` runs the judge on each case, prints a
# report, and writes it to the configured sink (Confident
# AI if connected, else the local cache). The case PASSES
# for this classifier when the returned label matches the
# expected one.

test_case_matches_user = LLMTestCase(
    input="¿Dónde está mi pedido?",
    actual_output=(
        "Su pedido fue enviado ayer y llegará el jueves."
    ),
    expected_labels={classifier.name: "matches_user"},
)

test_case_mismatch = LLMTestCase(
    input="¿Dónde está mi pedido?",
    actual_output=(
        "Your order was shipped yesterday and will arrive "
        "on Thursday."
    ),
    expected_labels={classifier.name: "mismatch"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# Both labels are exercised in the same run, so the
# match rate and the mismatch rate come from the same
# judge and the same scale. That's the value of testing
# language adherence through a fixed label set: drift
# back toward English (or toward the system prompt's
# language) becomes a measurable regression rather than
# a vibes-based review.

evaluate(
    test_cases=[test_case_matches_user, test_case_mismatch],
    classifiers=[classifier],
)