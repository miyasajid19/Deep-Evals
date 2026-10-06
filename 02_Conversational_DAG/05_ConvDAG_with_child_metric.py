from basemodel import CustomOpenAI
from deepeval.metrics.conversational_dag import (
    ConversationalBinaryJudgementNode,
)
from deepeval.metrics.dag import DeepAcyclicGraph
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams
from deepeval.metrics import ConversationalDAGMetric, ConversationalGEval
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE A CHILD ConversationalGEval
# ==========================================
# When a verdict's `then=` points to a BaseConversationalMetric, the child
# metric's score becomes the ConversationalDAGMetric's result for that path.
# Useful when a hard gate filters out clear failures, and the remaining
# conversations need a richer evaluation.

politeness = ConversationalGEval(
    model=CustomOpenAI(),
    name="Politeness",
    criteria="Determine whether the assistant remains polite throughout the conversation.",
    evaluation_steps=[
        "Check that every assistant turn maintains a courteous tone.",
        "Heavily penalise any sarcasm, dismissiveness, or rude language.",
        "Rewarding responses that stay respectful even with difficult users is OK.",
    ],
    evaluation_params=[MultiTurnParams.CONTENT],
)

# ==========================================
# 2. DEFINE THE DAG
# ==========================================
# Binary gate first: was the user's question answered?
# - False → score 0 immediately (no point judging tone).
# - True  → delegate to the ConversationalGEval child metric.

answered = ConversationalBinaryJudgementNode(
    criteria="Did the assistant answer the user's question?",
    evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
)

answered.add_verdict(verdict=False, score=0)
answered.add_verdict(verdict=True, then=politeness)

# ==========================================
# 3. BUILD DAG METRIC
# ==========================================
metric = ConversationalDAGMetric(
    name="Answered + Politeness",
    dag=DeepAcyclicGraph(root_nodes=[answered]),
    model=CustomOpenAI(),
    threshold=0.5,
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 4. CONVERSATIONAL TEST CASES
# ==========================================
test_cases = [
    # Answered + polite → child metric decides the score
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="What's the capital of France?"),
            Turn(role="assistant", content="The capital of France is Paris — happy to help if you have any follow-up questions!"),
        ]
    ),
    # Answered + rude → child metric scores low
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="What's the capital of France?"),
            Turn(role="assistant", content="It's Paris. Obviously. Try Google next time."),
        ]
    ),
    # Not answered → score 0 immediately
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="What's the capital of France?"),
            Turn(role="assistant", content="I'm not sure, maybe Berlin?"),
        ]
    ),
]

# ==========================================
# 5. EVALUATE
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])