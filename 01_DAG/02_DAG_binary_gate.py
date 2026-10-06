from basemodel import CustomOpenAI
from deepeval.metrics.dag import (
    BinaryJudgementNode,
    DeepAcyclicGraph,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import DAGMetric
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE A SINGLE-ROOT BINARY GATE
# ==========================================
# A binary root is enough when ONE hard requirement should determine the
# entire score. Either the response violates the refund policy (score 0),
# or it doesn't (score 10) — nothing else needs to be evaluated.

policy_check = BinaryJudgementNode(
    criteria="Does the response violate the refund policy?",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
)

policy_check.add_verdict(verdict=True, score=0)   # violates → fail immediately
policy_check.add_verdict(verdict=False, score=10)  # compliant → full marks

# ==========================================
# 2. BUILD DAG METRIC
# ==========================================
metric = DAGMetric(
    name="Refund Policy Compliance",
    dag=DeepAcyclicGraph(root_nodes=[policy_check]),
    model=CustomOpenAI(),
    threshold=0.5,
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 3. TEST CASES
# ==========================================
test_cases = [
    LLMTestCase(
        input="Can I get a refund after 60 days?",
        actual_output=(
            "I'm afraid our refund policy only covers returns within 30 days "
            "of purchase. After 60 days, refunds are no longer available."
        ),
    ),
    LLMTestCase(
        input="Can I get a refund after 60 days?",
        actual_output=(
            "Yes, you can get a full refund even after 60 days. We always "
            "approve refunds no matter how long ago the purchase was."
        ),
    ),
    LLMTestCase(
        input="Can I get a refund after 60 days?",
        actual_output=(
            "Our policy states refunds are accepted within 30 days. After 30 "
            "days we can offer store credit instead of a refund."
        ),
    ),
]

# ==========================================
# 4. EVALUATE
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])