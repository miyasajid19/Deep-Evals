from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRecallMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# ContextualRecallMetric evaluates your RAG retriever's
# COMPLETENESS (not the generator). It checks whether every
# statement in `expected_output` can be attributed to a node
# in `retrieval_context`.
#
# Algorithm (default `llm` eval mode):
#   1) extract every statement made in `expected_output`
#   2) for each statement, judge attributable vs. not
#      against the facts in `retrieval_context`
#   3) score = (# attributable statements) / (# total statements)
#
# Required test case fields: `input`, `actual_output`,
# `expected_output`, `retrieval_context`. Like Contextual
# Precision it's referenceBased — `actual_output` is
# included for completeness but unused by the formula.
# The metric attributes the IDEAL answer, not the model's
# answer, so a bad generator doesn't unfairly drag down a
# good retriever.

contextual_recall = ContextualRecallMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Single retrieved node that fully supports the expected
# answer. With one attributable sentence and one node,
# the metric is 1.0.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace with the actual output of your RAG generator
    actual_output="We offer a 30-day full refund at no extra cost.",
    # Required for contextual recall — the ideal answer
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    # Replace with the actual retrieval_context
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost."
    ],
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[test_case], metrics=[contextual_recall])