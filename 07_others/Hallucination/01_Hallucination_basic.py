from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# HallucinationMetric checks whether your LLM's
# `actual_output` introduces any CONTRADICTIONS against
# the supplied `context`. Higher is better.
#
# Algorithm (default `llm` eval mode):
#   1) for each context string, the LLM classifies
#      whether the actual_output contradicts it
#      (aligned = no contradiction).
#   2) score = number of contexts the output stayed
#             consistent with /
#             total number of contexts
#
# IMPORTANT: `context` is treated as ground-truth
# reference material — different from `retrieval_context`
# which is what your retriever fetched. This metric is
# for evaluating the generator against a trusted source,
# NOT for evaluating a live RAG system (use
# FaithfulnessMetric for that).
#
# This metric is `referenceBased` — it requires
# `context`.

# ==========================================
# 2. THE TEST CASE
# ==========================================
# The context describes a blond man drinking from a
# public water fountain. The actual_output says
# "A blond drinking water in public." — fully aligned.
# Expect score 1.0.

hallucination = HallucinationMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

test_case = LLMTestCase(
    input="What was the blond doing?",
    actual_output="A blond drinking water in public.",
    context=[
        "A man with blond-hair, and a brown shirt drinking out of a public water fountain.",
    ],
)

# ==========================================
# 3. RUN THE EVAL
# =========================================_
# `evaluate()` runs the metric, prints results, and
# writes a report (Confident AI if configured, else
# local cache).

evaluate(test_cases=[test_case], metrics=[hallucination])