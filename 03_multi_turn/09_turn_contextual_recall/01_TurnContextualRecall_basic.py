from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRecallMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# TurnContextualRecallMetric evaluates the COMPLETENESS
# of retrieval across a multi-turn RAG conversation. It's
# the multi-turn sibling of ContextualRecallMetric —
# instead of one (input, retrieval_context) tuple, you
# supply a `ConversationalTestCase` where each Turn
# carries its own retrieval_context.
#
# Algorithm (default `llm` eval mode):
#   1) build sliding windows of turns (size = `window_size`,
#      default 10)
#   2) for each window:
#        - break the `expected_outcome` into statements
#        - judge each statement attributable vs. not
#          against the window's retrieval_context
#   3) score = mean(per-window recall)
#
# `expected_outcome` is required on the test case — the
# metric attributes the IDEAL outcome's statements to
# retrieved nodes, so a good retriever / bad generator
# combination doesn't unfairly drag the score down. The
# metric is `referenceBased`; no `expected_output` per
# turn needed.

turn_contextual_recall = TurnContextualRecallMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# One assistant turn whose retrieval_context fully
# supports the expected outcome. Single attributable
# statement, perfect score.

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

evaluate(test_cases=[convo_test_case], metrics=[turn_contextual_recall])