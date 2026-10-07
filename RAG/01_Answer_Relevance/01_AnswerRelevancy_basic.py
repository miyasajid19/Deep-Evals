from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# AnswerRelevancyMetric is `referenceless` and `singleTurn`:
# it only needs `input` + `actual_output`. There's no
# `expected_output` or `retrieval_context` to set.
#
# The judge works in two steps:
#   1) extract every statement from `actual_output`
#   2) classify each statement as relevant to `input`
# score = (# relevant statements) / (# total statements)
#
# Default eval mode is `llm` (LLM-as-a-judge). The metric is
# self-explaining — passing `include_reason=True` makes the LLM
# emit the reason for its score.

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# A RAG-style Q&A where the answer is on-topic and brief.
# Every statement addresses the question, so we expect a high relevancy.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the output from your RAG generator
    actual_output="We offer a 30-day full refund at no extra cost.",
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[test_case], metrics=[answer_relevancy])