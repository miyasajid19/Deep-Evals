from basemodel import CustomOpenAI
from deepeval.metrics.conversational_dag import (
    ConversationalBinaryJudgementNode,
)
from deepeval.metrics.dag import DeepAcyclicGraph
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams
from deepeval.metrics import ConversationalDAGMetric
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE A SINGLE-ROOT BINARY GATE
# ==========================================
# A binary root is enough when ONE hard requirement over the whole
# conversation should determine the score. Either the user's issue was
# resolved by the end of the conversation (score 10), or it wasn't
# (score 0) — nothing else needs to be evaluated.

resolution_check = ConversationalBinaryJudgementNode(
    criteria="Was the user's issue resolved by the end of the conversation?",
    evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
    # turn_window=(0,7), # to prevent model to evaluate the conversation after the 7th turn, which is the last turn in the test cases, this will raise error if other testcases underflow in it
)

resolution_check.add_verdict(verdict=False, score=0)
resolution_check.add_verdict(verdict=True, score=10)

# ==========================================
# 2. BUILD DAG METRIC
# ==========================================
metric = ConversationalDAGMetric(
    name="Resolution Check",
    dag=DeepAcyclicGraph(root_nodes=[resolution_check]),
    model=CustomOpenAI(),
    threshold=0.5,
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 3. CONVERSATIONAL TEST CASES
# ==========================================
test_cases = [
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="My package never arrived. Can you help?"),
            Turn(role="assistant", content="I'm sorry to hear that. Could you share the order number?"),
            Turn(role="user", content="It's order #12345."),
            Turn(role="assistant", content="Thanks. I've located the shipment and dispatched a replacement — it will arrive in two business days."),
            Turn(role="user", content="Great, thank you!"),
            Turn(role="assistant", content="You're welcome! Is there anything else I can assist you with?"),
            Turn(role="user", content="No, that's all. Thanks again!"),
            Turn(role="assistant", content="My pleasure! Have a great day!"),
            Turn(role="user", content="You too!"),
            Turn(role="assistant", content="Thank you! Goodbye!"),
        ]
    ),
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="My package never arrived. Can you help?"),
            Turn(role="assistant", content="I'm sorry to hear that. Could you share the order number?"),
            Turn(role="user", content="It's order #12345."),
            Turn(role="assistant", content="Thanks. Let me check on that and get back to you shortly."),
            Turn(role="user", content="Okay, I'll wait."),
        ]
    ),
]

# ==========================================
# 4. EVALUATE
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])