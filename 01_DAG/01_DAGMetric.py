from basemodel import CustomOpenAI
from deepeval.metrics.dag import (
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    DeepAcyclicGraph,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import DAGMetric
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE DAG NODES
# ==========================================

# Root node — checks if the answer is factually correct
correctness = BinaryJudgementNode(
    criteria="Is the actual output factually correct for the given input?",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
)

# Child node — grades quality of correct answers
response_quality = NonBinaryJudgementNode(
    criteria="Classify the quality of the correct response.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
)

# If correct → grade quality; if incorrect → score 0 immediately
correctness.add_verdict(verdict=True, then=response_quality)
correctness.add_verdict(verdict=False, score=0)

# Quality scoring
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
# 3. NARUTO-THEMED TEST CASES
# ==========================================
test_cases = [
    LLMTestCase(
        input="Who is the Fourth Hokage in Naruto?",
        actual_output="The Fourth Hokage is Minato Namikaze, also known as the Yellow Flash of the Leaf."
    ),
    LLMTestCase(
        input="What is sealed inside Naruto Uzumaki?",
        actual_output="Kurama, the Nine-Tails beast, is sealed inside Naruto Uzumaki."
    ),
    LLMTestCase(
        input="What village does Sasuke Uchiha belong to?",
        actual_output="Sasuke Uchiha belongs to the Hidden Sand Village."  # Wrong answer intentionally
    ),
]

# ==========================================
# 4. EVALUATE
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])