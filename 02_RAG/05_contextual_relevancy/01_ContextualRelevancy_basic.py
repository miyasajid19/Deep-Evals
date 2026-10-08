from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# ContextualRelevancyMetric measures SIGNAL-VS-NOISE in
# your RAG retriever's `retrieval_context`. It's the
# retriever-only sibling of `AnswerRelevancyMetric` (which
# judges actual_output) and `FaithfulnessMetric` (which
# judges actual_output against retrieval_context).
#
# Algorithm (default `llm` eval mode):
#   1) extract every statement from `retrieval_context`
#   2) for each statement, judge relevant vs. irrelevant
#      against `input`
#   3) score = (# relevant statements) / (# total statements)
#
# Required test case fields: `input`, `actual_output`,
# `retrieval_context`. The metric is `referenceless`: no
# `expected_output` needed. It grades the RETRIEVER, not
# the generator — irrelevant nodes hurt the score even
# when the answer is right.

contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Single relevant retrieval node. Single statement, fully
# on-topic for the input. Expect score 1.0.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your RAG generator
    actual_output="We offer a 30-day full refund at no extra cost.", # it is not necessary to include as contextual relevancy is compared with input vs retrieval_context, not actual_output
    # Replace this with the actual retrieval_context
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost."
    ],
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[test_case], metrics=[contextual_relevancy])