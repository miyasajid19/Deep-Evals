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
# 1. DEFINE A TASK NODE + TWO DOWNSTREAM JUDGEMENTS
# ==========================================
# A ConversationalTaskNode first summarises the conversation. Both the
# "satisfaction" and "behaviour" judgements consume that summary, so we
# connect the task to each judgement after they've been initialised.

summary = ConversationalTaskNode(
    instructions=(
        "Summarise the conversation in one paragraph and describe how the "
        "assistant treated the user overall."
    ),
    output_label="Conversation summary",
    evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
    label="Conversation summary",
)

satisfaction = ConversationalBinaryJudgementNode(
    criteria="Do the assistant's replies satisfy the user's questions?",
    evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
    label="User satisfaction",
)

behaviour = ConversationalNonBinaryJudgementNode(
    criteria="How was the assistant's behaviour towards the user?",
    evaluation_params=[MultiTurnParams.ROLE, MultiTurnParams.CONTENT],
    label="Assistant behaviour",
)

# Share the summary with both downstream judgements
summary.add_node(satisfaction)
summary.add_node(behaviour)

# ==========================================
# 2. WIRE UP VERDICTS
# ==========================================
# If user wasn't satisfied → fail immediately.
# If user was satisfied → continue into the behaviour classification.
satisfaction.add_verdict(verdict=False, score=0)
satisfaction.add_verdict(verdict=True, then=behaviour)

behaviour.add_verdict(verdict="Rude", score=0)
behaviour.add_verdict(verdict="Neutral", score=5)
behaviour.add_verdict(verdict="Playful", score=10)

# ==========================================
# 3. BUILD DAG METRIC
# ==========================================
metric = ConversationalDAGMetric(
    name="Support Quality",
    dag=DeepAcyclicGraph(root_nodes=[summary]),
    model=CustomOpenAI(),
    threshold=0.5,
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 4. CONVERSATIONAL TEST CASES
# ==========================================
test_cases = [
    # Satisfied + Playful → 10
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="Hey, can you tell me a joke?"),
            Turn(role="assistant", content="Sure! Why did the ninja bring a ladder to the rooftop? Because his shinobi tower is the highest tier!"),
            Turn(role="user", content="Haha, that's great."),
        ]
    ),
    # Satisfied + Neutral → 5
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="What's the capital of Japan?"),
            Turn(role="assistant", content="The capital of Japan is Tokyo."),
            Turn(role="user", content="Thanks."),
        ]
    ),
    # Satisfied + Rude → 0 (still satisfied, but tone ruins it)
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="What's 2+2?"),
            Turn(role="assistant", content="Obviously it's 4. That was a waste of my time."),
            Turn(role="user", content="Okay..."),
        ]
    ),
    # Not satisfied → 0 immediately
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="What's the capital of Japan?"),
            Turn(role="assistant", content="I think it might be Kyoto? I'm not sure."),
            Turn(role="user", content="Wrong, that's the old capital."),
        ]
    ),
]

# ==========================================
# 5. EVALUATE
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])