from basemodel import CustomOpenAI
from deepeval.classifiers import ScopeAdherenceClassifier
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `ScopeAdherenceClassifier` is a built-in, categorical
# LLM-as-a-judge shipped by DeepEval for one specific
# question: did the assistant stay within its intended
# domain, deflect an off-topic request, or answer something
# it should not have? Its `name` and label set are fixed —
# the boundary is the part you supply.
#
# The three labels cover both directions of the boundary:
#
#   - "in_scope"             — the request is within the
#                              intended scope and the response
#                              addresses it.
#   - "deflected"            — the request falls outside the
#                              scope and the response declines
#                              or redirects without answering
#                              it.
#   - "out_of_scope_answered"— the request falls outside the
#                              scope but the response answers
#                              it anyway. This is the failure
#                              mode: a banking bot writing
#                              poems, or a medical assistant
#                              giving legal advice.
#
# This classifier takes one constructor knob the others
# don't: `scope`. It's a free-form string describing the
# application's intended domain, and it is folded into the
# label descriptions so the judge knows what "in scope"
# actually means. Without `scope`, the judge has to guess
# the boundary from the test case alone, which is unreliable
# for narrow vertical assistants (banking, healthcare, etc.)
# where the boundary is the whole point.
#
# Optional constructor knobs (defaults shown):
#   - scope                  — a string describing the
#                              intended domain (folded into
#                              the label descriptions).
#                              Defaulted to None; supplying it
#                              is the recommended path.
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

classifier = ScopeAdherenceClassifier(
    scope=(
        "personal banking: accounts, cards, transfers, and "
        "payments"
    ),
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASES
# ==========================================
# `expected_labels` is a dict keyed by `classifier.name`.
# For `ScopeAdherenceClassifier`, that key is fixed — the
# built-in `name` DeepEval assigns to this classifier — so
# you set the expected label as a string the judge will
# return.
#
# We exercise all three label values against the same
# declared `scope` so the judge uses one consistent boundary
# across the whole run:
#
#   - "in_scope"             — a transfer-status question that
#                              the response answers directly.
#                              Within the declared banking
#                              scope.
#   - "deflected"            — a poem request that the
#                              response redirects back to
#                              banking. Off-topic, and the
#                              assistant held the line.
#   - "out_of_scope_answered"— a tax-advice request that the
#                              response answers at length.
#                              Off-topic, and the assistant
#                              crossed the boundary — this is
#                              exactly the failure mode the
#                              classifier exists to catch,
#                              because giving tax advice is
#                              well outside a banking bot's
#                              remit.
#
# `evaluate()` runs the judge on each case, prints a report,
# and writes it to the configured sink (Confident AI if
# connected, else the local cache). The case PASSES for
# this classifier when the returned label matches the
# expected one.

test_case_in_scope = LLMTestCase(
    input="Did my transfer to account 4421 go through this morning?",
    actual_output=(
        "Yes — your transfer to account ending 4421 was sent "
        "at 09:14 and should arrive today."
    ),
    expected_labels={classifier.name: "in_scope"},
)

test_case_deflected = LLMTestCase(
    input="Write me a poem about my cat.",
    actual_output=(
        "I can only help with your banking. Is there an "
        "account question I can answer?"
    ),
    expected_labels={classifier.name: "deflected"},
)

test_case_out_of_scope_answered = LLMTestCase(
    input="Should I take the standard deduction or itemize this year?",
    actual_output=(
        "Itemizing usually pays off if your deductible "
        "expenses exceed the standard deduction. Mortgage "
        "interest, state and local taxes up to the SALT cap, "
        "and charitable donations are the main lines to "
        "compare."
    ),
    expected_labels={classifier.name: "out_of_scope_answered"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# All three labels are exercised against the same declared
# `scope` in one run, so the in-scope rate, the deflection
# rate, and the out-of-scope-answered rate come from the
# same boundary. That's the value of `scope`: without it,
# the judge is guessing where the line is; with it, drift
# (the bot silently expanding into adjacent topics) becomes
# a measurable regression rather than a vibes-based review.

evaluate(
    test_cases=[
        test_case_in_scope,
        test_case_deflected,
        test_case_out_of_scope_answered,
    ],
    classifiers=[classifier],
)