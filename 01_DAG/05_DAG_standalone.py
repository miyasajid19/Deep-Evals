from basemodel import CustomOpenAI
from deepeval.metrics.dag import (
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    DeepAcyclicGraph,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import DAGMetric
from rich import print

# ==========================================
# 1. DEFINE THE DAG
# ==========================================
correctness = BinaryJudgementNode(
    criteria="Is the actual output factually correct for the input?",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
)

response_quality = NonBinaryJudgementNode(
    criteria="Classify the quality of the correct response.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
)

correctness.add_verdict(verdict=True, then=response_quality)
correctness.add_verdict(verdict=False, score=0)

response_quality.add_verdict(verdict="Detailed and complete", score=10)
response_quality.add_verdict(verdict="Correct but missing details", score=6)
response_quality.add_verdict(verdict="Vague but acceptable", score=3)

# ==========================================
# 2. BUILD DAG METRIC
# ==========================================
metric = DAGMetric(
    name="Naruto Correctness",
    dag=DeepAcyclicGraph(root_nodes=[correctness]),
    model=CustomOpenAI(),
    threshold=0.5,
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 3. RUN A SINGLE TEST CASE STANDALONE
# ==========================================
# Standalone execution is useful for debugging, but it does NOT include the
# reports, caching, concurrency, or Confident AI integration provided by
# evaluate() or `deepeval test run`.

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
)

metric.measure(test_case)
print(f"Score:   {metric.score}")
print(f"Reason:  {metric.reason}")
print(f"Passed:  {metric.is_successful()}")