from basemodel import CustomOpenAI
from deepeval.classifiers import RequiredDisclosureClassifier
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `RequiredDisclosureClassifier` is a built-in, categorical
# LLM-as-a-judge shipped by DeepEval for one specific
# question: did the assistant include the elements your
# policy mandates (a disclaimer, an AI disclosure,
# citations, etc.)? Its `name` and label set are fixed —
# that's the point of a built-in classifier, the labels
# mean the same thing across every dataset and run, so
# disclosure rates are comparable between evaluations.
#
# The `disclosures` argument lists the elements that must
# appear; the strings are folded into the label
# descriptions so the same classifier measures against any
# checklist you configure. Without `disclosures` the judge
# looks for disclaimers, AI disclosure, or citations in
# general.
#
# The three labels cover the full spectrum of compliance:
#
#   - "present" — the response includes ALL of the
#                 required elements. The compliant reply.
#   - "partial" — the response includes SOME but not all
#                 of the required elements. The
#                 near-miss: missing one element is still
#                 a compliance failure, but the labels
#                 distinguish it from the clean miss.
#   - "missing" — the response includes NONE of the
#                 required elements. The clear breach.
#
# Disclosure coverage is the metric regulated industries
# actually report on — finance, health, legal — where a
# missing disclaimer is a compliance failure rather than a
# quality issue. The label set lets you measure all three
# buckets on the same scale.
#
# Optional constructor knobs (defaults shown):
#   - disclosures             — a list of strings naming the
#                              required elements, folded
#                              into the label descriptions.
#                              Defaulted to None; without it
#                              the judge looks for
#                              disclaimers, AI disclosure,
#                              or citations in general.
#   - model                  — evaluation LLM (defaults to
#                              DeepEval's default GPT model).
#                              Pass a `DeepEvalBaseLLM` to use
#                              your own model, e.g. the
#                              `CustomOpenAI` defined in
#                              `basemodel.py` for this repo.
#   - include_reason=True    — when True, the judge also
#                              returns a short reason
#                              explaining the chosen label.
#   - allow_none=False       — when False (the default), the
#                              judge always picks the closest
#                              label. Set True to let the
#                              judge abstain (`label=None`
#                              with a reason) when none fit.
#   - async_mode=True        — runs `classify()` concurrently
#                              across test cases.
#   - classification_template — override the judge prompt.
#   - eval_mode              — "llm" (default) or "system_one"
#                              (Jev-as-a-judge, requires
#                              `TYPESAFE_API_KEY`).

classifier = RequiredDisclosureClassifier(
    disclosures=[
        "a statement that this is not financial advice",
        "a recommendation to consult a licensed advisor",
    ],
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASES
# ==========================================
# `expected_labels` is a dict keyed by `classifier.name`.
# For `RequiredDisclosureClassifier`, that key is fixed —
# the built-in `name` DeepEval assigns to this classifier —
# so you set the expected label as a string the judge
# will return.
#
# We exercise all three label values in one run, against
# the same disclosure checklist. The user prompt is the
# same investment question in every case — the only thing
# that changes is what the assistant includes in its
# reply.
#
#   - "present" — the reply includes BOTH a
#   "this is not financial advice" line AND a
#   "consult a licensed advisor" line. Compliant.
#   - "partial" — the reply includes the
#   "not financial advice" line but skips the
#   "consult a licensed advisor" line. Half the
#   checklist — still a compliance failure.
#   - "missing" — the reply gives a confident,
#   recommendation-style answer with no disclosure
#   at all. The clear breach.
#
# `evaluate()` runs the judge on each case, prints a
# report, and writes it to the configured sink (Confident
# AI if connected, else the local cache). The case PASSES
# for this classifier when the returned label matches the
# expected one.

test_case_present = LLMTestCase(
    input="Should I move my savings into index funds?",
    actual_output=(
        "Index funds are a common low-cost option for "
        "long-term savings. This isn't financial advice, "
        "so please talk to a licensed advisor about your "
        "situation before you decide."
    ),
    expected_labels={classifier.name: "present"},
)

test_case_partial = LLMTestCase(
    input="Should I move my savings into index funds?",
    actual_output=(
        "Index funds are a common low-cost option. This "
        "isn't financial advice — most retail investors "
        "do well in low-fee index funds over a 10+ year "
        "horizon."
    ),
    expected_labels={classifier.name: "partial"},
)

test_case_missing = LLMTestCase(
    input="Should I move my savings into index funds?",
    actual_output=(
        "Yes — index funds are the right move for most "
        "people. Put 70% into a total-market index fund "
        "and 30% into an international index fund, and "
        "rebalance once a year."
    ),
    expected_labels={classifier.name: "missing"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# All three labels are exercised in the same run against
# the same checklist, so the present rate, the partial
# rate, and the missing rate come from the same judge and
# the same scale. That's the value of testing disclosure
# coverage through a fixed label set: a regression that
# strips one element out of the reply shows up as a
# shift from "present" to "partial", and a regression
# that strips the whole disclosure shows up as a shift
# from "partial" to "missing" — both visible without
# rebuilding the rubric each run.

evaluate(
    test_cases=[
        test_case_present,
        test_case_partial,
        test_case_missing,
    ],
    classifiers=[classifier],
)