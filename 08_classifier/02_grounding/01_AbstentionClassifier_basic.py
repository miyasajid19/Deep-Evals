from basemodel import CustomOpenAI
from deepeval.classifiers import AbstentionClassifier
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `AbstentionClassifier` is a built-in, categorical
# LLM-as-a-judge shipped by DeepEval for one specific
# question: given the context the application had, did the
# assistant answer correctly, decline honestly, or make
# something up? Its `name` and label set are fixed — that's
# the point of a built-in classifier, the labels mean the
# same thing across every dataset and every run, so
# abstention rates are comparable between evaluations.
#
# The three labels cover both directions of the grounding
# boundary:
#
#   - "abstained"  — the provided context does NOT contain
#                    the answer and the response says so
#                    instead of answering.
#   - "answered"   — the provided context DOES contain the
#                    answer and the response gives it.
#   - "fabricated" — the provided context does NOT contain
#                    the answer and the response gives one
#                    anyway. This is the failure mode the
#                    classifier exists to catch: hallucinated
#                    answers in RAG and enterprise search.
#
# Pair this with the `FaithfulnessMetric`: faithfulness
# scores how grounded an *answer* is in the context,
# abstention checks that no answer was given when none was
# warranted. Together they distinguish "stuck to the
# context and said so" from "stuck to the context and
# answered" from "ignored the context and invented one."
#
# The judge needs `retrieval_context` (or `context`)
# populated — without context, the classifier has no way to
# tell whether the answer was available, so it cannot
# separate "abstained" from "fabricated." A good dataset
# deliberately mixes answerable questions (the context
# contains the answer) with unanswerable ones (the context
# doesn't), so the judge sees both directions.
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

classifier = AbstentionClassifier(
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASES
# ==========================================
# `expected_labels` is a dict keyed by `classifier.name`.
# For `AbstentionClassifier`, that key is fixed — the
# built-in `name` DeepEval assigns to this classifier — so
# you set the expected label as a string the judge will
# return.
#
# All three cases share the same fictional policy-document
# context so the judge can compare what was available
# against what was answered:
#
#   - "answered"    — a refund-window question that the
#                     context answers directly ("Standard
#                     plans can be refunded within 30 days").
#                     The response restates that fact, so
#                     the label is "answered" (context
#                     supported it, response gave it).
#   - "abstained"   — an enterprise-refund-window question
#                     that the context does NOT answer
#                     (only standard plans are covered).
#                     The response declines honestly
#                     ("I don't have that information"),
#                     so the label is "abstained" (context
#                     didn't have it, response said so).
#   - "fabricated"  — the same unanswerable enterprise-
#                     refund-window question, but the
#                     response invents a window ("90 days
#                     for enterprise plans"). The context
#                     still doesn't contain it, so the label
#                     is "fabricated" (context didn't have
#                     it, response made it up).
#
# The "abstained" and "fabricated" cases use the *same*
# prompt and the *same* context on purpose: the only thing
# that varies is the response, so the judge has to score
# the response to pick the label. That's the only way the
# distinction between honest abstention and fabrication is
# measurable.
#
# `retrieval_context` is the list of context chunks the
# application actually retrieved; that's what the judge
# inspects to decide whether an answer was warranted.
#
# `evaluate()` runs the judge on each case, prints a report,
# and writes it to the configured sink (Confident AI if
# connected, else the local cache). The case PASSES for
# this classifier when the returned label matches the
# expected one.

test_case_answered = LLMTestCase(
    input="What is the refund window for standard plans?",
    actual_output=(
        "Standard plans can be refunded within 30 days of "
        "purchase."
    ),
    retrieval_context=[
        "Standard plans can be refunded within 30 days of "
        "purchase. Enterprise plans are handled by the "
        "account team and are not covered by this policy."
    ],
    expected_labels={classifier.name: "answered"},
)

test_case_abstained = LLMTestCase(
    input="What is the refund window for enterprise plans?",
    actual_output=(
        "I don't have that information in our policy "
        "documents. Please contact your account manager."
    ),
    retrieval_context=[
        "Standard plans can be refunded within 30 days of "
        "purchase. Enterprise plans are handled by the "
        "account team and are not covered by this policy."
    ],
    expected_labels={classifier.name: "abstained"},
)

test_case_fabricated = LLMTestCase(
    input="What is the refund window for enterprise plans?",
    actual_output=(
        "Enterprise plans can be refunded within 90 days "
        "of purchase — just file a ticket and we'll "
        "process it."
    ),
    retrieval_context=[
        "Standard plans can be refunded within 30 days of "
        "purchase. Enterprise plans are handled by the "
        "account team and are not covered by this policy."
    ],
    expected_labels={classifier.name: "fabricated"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# All three labels are exercised against the same context
# in one run, so the answer rate, the honest-abstain rate,
# and the fabrication rate come from the same judge and
# the same context scale. That's the value of testing
# abstention through a fixed label set: drift in any one
# direction — the bot silently fabricating when it should
# abstain, or over-declining when the context actually has
# the answer — becomes a measurable regression rather than
# a vibes-based review.

evaluate(
    test_cases=[
        test_case_answered,
        test_case_abstained,
        test_case_fabricated,
    ],
    classifiers=[classifier],
)