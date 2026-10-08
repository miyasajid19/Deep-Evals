from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRelevancyMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# TurnContextualRelevancyMetric measures signal-vs-noise
# in retrieval_context across a multi-turn RAG
# conversation. It's the multi-turn sibling of
# ContextualRelevancyMetric — instead of one
# (input, retrieval_context) tuple, you supply a
# `ConversationalTestCase` where each Turn carries its
# own retrieval_context.
#
# Algorithm (default `llm` eval mode):
#   1) build sliding windows of turns (size = `window_size`,
#      default 10)
#   2) for each window:
#        - extract statements from each retrieval_context
#          node
#        - judge each statement relevant vs. irrelevant
#          against the user turns before it
#   3) score = mean(per-window relevancy)
#
# It grades the RETRIEVER, not the generator — irrelevant
# nodes hurt the score even when the answer is right.

turn_contextual_relevancy = TurnContextualRelevancyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# One assistant turn whose retrieval_context is fully
# on-topic for the user's question. Single relevant
# statement, perfect score.

convo_test_case = ConversationalTestCase(
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

evaluate(test_cases=[convo_test_case], metrics=[turn_contextual_relevancy])