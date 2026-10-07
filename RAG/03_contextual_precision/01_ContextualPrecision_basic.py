from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# ContextualPrecisionMetric evaluates your RAG retriever's
# RANKER (not the generator). It checks whether the nodes in
# `retrieval_context` that are useful for arriving at
# `expected_output` appear above the nodes that aren't.
#
# Algorithm (default `llm` eval mode):
#   1) for each node in `retrieval_context`, judge relevant
#      vs. irrelevant against (input, expected_output)
#   2) score = weighted cumulative precision (WCP) —
#      relevance at position k gets weighted by 1/k, so
#      a relevant node ranked early counts much more than
#      a relevant node ranked late
#
# Required test case fields: `input`, `actual_output`,
# `expected_output`, `retrieval_context`. It's the only
# RAG metric that REQUIRES `expected_output` — without
# an ideal answer there's no way to know which nodes were
# "useful".

contextual_precision = ContextualPrecisionMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Single retrieved node that exactly matches the expected
# answer. Order doesn't matter when there's only one
# node — the metric is trivially 1.0. The interesting
# cases are in `02_verbose` and `08_ranking_demo`.

test_case =[ LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace with the actual output of your RAG generator
    actual_output="We offer a 30-day full refund at no extra cost.",
    # Required for contextual precision — the ideal answer
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    # Replace with the actual retrieval_context
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost."
    ],
),
            LLMTestCase(
    input="What is your return policy?",
    actual_output="We offer a wide range of shoe sizes from 6 to 14.",
    expected_output="Customers are eligible for a 30-day full refund at no extra cost.",
    retrieval_context=[
        "Our shoe collection includes sizes ranging from 6 to 14.",
        "We stock over 200 different shoe styles for men and women."
    ]
            )
]
# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=test_case, metrics=[contextual_precision])