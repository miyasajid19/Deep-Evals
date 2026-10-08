from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualPrecisionMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# TurnContextualPrecisionMetric evaluates the RANKING of
# relevant retrieval_context nodes across a multi-turn
# RAG conversation. It's the multi-turn sibling of
# ContextualPrecisionMetric — instead of one
# (input, retrieval_context) tuple, you supply a
# `ConversationalTestCase` where each assistant Turn
# carries its own retrieval_context.
#
# Algorithm (default `llm` eval mode):
#   1) build sliding windows of turns (size = `window_size`,
#      default 10)
#   2) for each window:
#        - judge each retrieval_context node relevant vs.
#          irrelevant against `expected_outcome`
#        - compute weighted cumulative precision (WCP) —
#          position-weighted so relevant nodes ranked
#          earlier matter more
#   3) score = mean(per-window WCP)
#
# `expected_outcome` is required on the test case — it's
# the ground truth the metric uses to label each context
# node relevant or irrelevant. The metric is
# `referenceBased`; no `expected_output` per turn needed.

turn_contextual_precision = TurnContextualPrecisionMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# One assistant turn with a single relevant retrieval
# node. Order is trivially perfect. The interesting
# cases are in `02_verbose` and `06_ranking_demo`.

convo_test_case = ConversationalTestCase(
    expected_outcome=(
        "The chatbot must explain store policies clearly, including "
        "refund eligibility, return process, and shipping timelines."
    ),
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
        ),
    ]
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[convo_test_case], metrics=[turn_contextual_precision])