from basemodel import CustomOpenAI
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_span
from deepeval.metrics.dag import (
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    DeepAcyclicGraph,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import DAGMetric
from rich import print

# ==========================================
# 1. DEFINE A DAG FOR COMPONENT-LEVEL EVAL
# ==========================================
# Same pattern as a standalone DAG, but the metric is attached to an
# @observe-decorated component instead of being run via evaluate().

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

metric = DAGMetric(
    name="Naruto Correctness",
    dag=DeepAcyclicGraph(root_nodes=[correctness]),
    model=CustomOpenAI(),
    threshold=0.5,
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. SIMULATED LLM CALL
# ==========================================
def call_llm(input: str) -> str:
    responses = {
        "Who is the Fourth Hokage?": (
            "The Fourth Hokage is Minato Namikaze, also known as the "
            "Yellow Flash of the Leaf."
        ),
        "What is sealed inside Naruto?": (
            "Kurama, the Nine-Tails beast, is sealed inside Naruto Uzumaki."
        ),
        "What village is Naruto from?": (
            "Naruto is from the Hidden Leaf Village, known as Konohagakure."
        ),
    }
    return responses.get(input, "I don't know.")

# ==========================================
# 3. COMPONENT WITH DAG METRIC
# ==========================================
@observe(metrics=[metric])
def inner_component(input: str, expected_output: str):
    actual_output = call_llm(input)
    update_current_span(
        test_case=LLMTestCase(
            input=input,
            actual_output=actual_output,
            expected_output=expected_output,
        )
    )
    return actual_output

@observe()
def llm_app(input: str, expected_output: str):
    return inner_component(input, expected_output)

# ==========================================
# 4. DATASET
# ==========================================
dataset = EvaluationDataset(goldens=[
    Golden(
        input="Who is the Fourth Hokage?",
        expected_output="Minato Namikaze is the Fourth Hokage, also known as the Yellow Flash."
    ),
    Golden(
        input="What is sealed inside Naruto?",
        expected_output="Kurama, the Nine-Tails beast, is sealed inside Naruto Uzumaki."
    ),
    Golden(
        input="What village is Naruto from?",
        expected_output="Naruto is from the Hidden Leaf Village, known as Konohagakure."
    ),
])

# ==========================================
# 5. EVALUATE OVER THE DATASET
# ==========================================
for golden in dataset.evals_iterator():
    llm_app(golden.input, golden.expected_output)