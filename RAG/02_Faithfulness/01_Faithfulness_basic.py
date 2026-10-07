from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# FaithfulnessMetric checks whether the claims made in
# `actual_output` are supported by `retrieval_context`.
# It does NOT compare against ground truth — it only asks
# "does this answer contradict what was retrieved?".
#
# Algorithm (default `llm` eval mode):
#   1) extract every claim from `actual_output`
#   2) for each claim, judge faithful vs contradicts
#      against the facts in `retrieval_context`
#   3) score = (# truthful claims) / (# total claims)
#
# A claim with no information in the context to confirm or
# deny it counts as faithful (no contradiction found).
#
# Required test case fields: `input`, `actual_output`,
# `retrieval_context`. There is no `expected_output` and
# no ground-truth `context` — that's what the Hallucination
# metric is for.

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Customer asks the same shoe-fit question as in the
# AnswerRelevancy examples, but here the answer is
# grounded in retrieved policy text. Faithfulness should
# score 1.0 — the claim ("30-day full refund at no extra
# cost") matches the retrieved chunk.

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace with the actual output of your RAG generator
    actual_output="We offer a 30-day full refund at no extra cost.",
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

evaluate(test_cases=[test_case], metrics=[faithfulness])