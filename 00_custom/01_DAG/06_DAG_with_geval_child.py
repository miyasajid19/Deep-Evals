from basemodel import CustomOpenAI
from deepeval.metrics.dag import (
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    DeepAcyclicGraph,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import DAGMetric, GEval
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE A CHILD GEVAL METRIC
# ==========================================
# When a verdict's `then=` points to a BaseMetric, the child metric's score
# becomes the DAGMetric's result for that path. Useful when a hard gate
# filters out clear failures, and the remaining cases need a richer
# evaluation.

writing_quality = GEval(
    model=CustomOpenAI(),
    name="Writing Quality",
    criteria="Determine whether the response is clear, concise, and well written.",
    evaluation_steps=[
        "Check that the response communicates the intended information clearly.",
        "Penalize vague, run-on, or repetitive sentences.",
        "Reward responses that are concise and easy to follow.",
    ],
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
)

# ==========================================
# 2. DEFINE THE DAG
# ==========================================
# Binary gate first: is the response factually correct?
# - False → score 0 immediately (no need to judge writing quality).
# - True  → delegate to the GEval child metric for a finer score.

correctness = BinaryJudgementNode(
    criteria="Is the actual output factually correct for the input?",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
)

correctness.add_verdict(verdict=False, score=0)
correctness.add_verdict(verdict=True, then=writing_quality)

# ==========================================
# 3. BUILD DAG METRIC
# ==========================================
metric = DAGMetric(
    name="Correctness + Writing Quality",
    dag=DeepAcyclicGraph(root_nodes=[correctness]),
    model=CustomOpenAI(),
    threshold=0.5,
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 4. TEST CASES
# ==========================================
test_cases = [
    LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
    ),
    LLMTestCase(
        input="What is the capital of France?",
        actual_output=(
            "France has many important cities, but if you are asking for the "
            "capital, the answer is Paris."
        ),
    ),
    LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Berlin.",
    ),
]

# ==========================================
# 5. EVALUATE
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])