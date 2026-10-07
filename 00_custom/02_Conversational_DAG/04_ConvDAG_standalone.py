from basemodel import CustomOpenAI
from deepeval.metrics.conversational_dag import (
    ConversationalBinaryJudgementNode,
)
from deepeval.metrics.dag import DeepAcyclicGraph
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams
from deepeval.metrics import ConversationalDAGMetric
from rich import print

# ==========================================
# 1. DEFINE THE DAG
# ==========================================
satisfaction = ConversationalBinaryJudgementNode(
    criteria="Do the assistant's replies satisfy the user's questions?",
    evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
)

satisfaction.add_verdict(verdict=True, score=10)
satisfaction.add_verdict(verdict=False, score=0)

# ==========================================
# 2. BUILD DAG METRIC
# ==========================================
metric = ConversationalDAGMetric(
    name="User Satisfaction",
    dag=DeepAcyclicGraph(root_nodes=[satisfaction]),
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

test_case = ConversationalTestCase(
    turns=[
        Turn(role="user", content="What's the weather in Paris?"),
        Turn(role="assistant", content="Sunny and 24°C today."),
    ]
)

metric.measure(test_case)
print(f"Score:   {metric.score}")
print(f"Reason:  {metric.reason}")
print(f"Passed:  {metric.is_successful()}")