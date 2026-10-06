from basemodel import CustomOpenAI
from deepeval.metrics.conversational_dag import (
    ConversationalTaskNode,
    ConversationalBinaryJudgementNode,
    ConversationalNonBinaryJudgementNode,
)
from deepeval.metrics.dag import DeepAcyclicGraph
from deepeval.test_case import ConversationalTestCase, Turn, MultiTurnParams
from deepeval.metrics import ConversationalDAGMetric
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE TURN-WINDOWED NODES
# ==========================================
# Two independent checks on different slices of the conversation:
#   - "greeting_check" looks at turns 0..1 (the opening exchange only).
#   - "closing_check" looks at turns 4..6 (the final exchange only).
# Each judgement receives its own slice via `turn_window=(start, end)`.

greeting_check = ConversationalBinaryJudgementNode(
    criteria="Did the assistant greet the user politely at the start?",
    evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
    turn_window=(0, 1),
    label="Greeting",
)

closing_check = ConversationalNonBinaryJudgementNode(
    criteria="How did the assistant close the conversation?",
    evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
    turn_window=(4, 5),
    label="Closing",
)

# ==========================================
# 2. WIRE UP BRANCHING
# ==========================================
# Both nodes start as roots (only TaskNode + JudgementNode as multiple
# roots is allowed). If the greeting fails, the path ends immediately
# at score 0. If the greeting passes, the closing judgement runs.

greeting_check.add_verdict(verdict=False, score=0)
greeting_check.add_verdict(verdict=True, then=closing_check)

closing_check.add_verdict(verdict="Friendly farewell", score=10)
closing_check.add_verdict(verdict="Abrupt sign-off", score=5)
closing_check.add_verdict(verdict="No closing", score=0)

# ==========================================
# 3. BUILD DAG METRIC
# ==========================================
metric = ConversationalDAGMetric(
    name="Conversation Etiquette",
    dag=DeepAcyclicGraph(root_nodes=[greeting_check]),
    model=CustomOpenAI(),
    threshold=0.5,
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 4. CONVERSATIONAL TEST CASES (6+ turns so turn windows fit)
# ==========================================
test_cases = [
    # Polite greeting + Friendly farewell → 10
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="Hi, can you help me reset my password?"),
            Turn(role="assistant", content="Hello! Of course, I'd be happy to help you reset your password."),
            Turn(role="user", content="Great, what's the first step?"),
            Turn(role="assistant", content="I'll send a reset link to your registered email."),
            Turn(role="user", content="Got it, thanks!"),
            Turn(role="assistant", content="You're welcome! Have a wonderful day and feel free to come back anytime."),
        ]
    ),
    # Polite greeting + Abrupt sign-off → 5  ===> this will cause error
    # ConversationalTestCase(
    #     turns=[
    #         Turn(role="user", content="Hi, can you help me reset my password?"),
    #         Turn(role="assistant", content="Hello! Of course, I'd be happy to help you reset your password."),
    #         Turn(role="user", content="Great, what's the first step?"),
    #         Turn(role="assistant", content="I'll send a reset link to your registered email."),
    #         Turn(role="user", content="Got it, thanks!"),
    #         Turn(role="assistant", content="Done."),
    #     ]
    # ),
    # Rude greeting → 0 immediately
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="Hi, can you help me reset my password?"),
            Turn(role="assistant", content="What do you want? Make it quick."),
            Turn(role="user", content="I'd like to reset my password."),
            Turn(role="assistant", content="Fine. Check your email."),
            Turn(role="user", content="Okay..."),
            Turn(role="assistant", content="Have a wonderful day and feel free to come back anytime."),
            Turn(role="user", content="Thanks, I guess."),
        ]
    ),
]

# ==========================================
# 5. EVALUATE
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])