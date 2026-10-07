from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import RoleAdherenceMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# RoleAdherenceMetric checks whether the assistant stays
# IN CHARACTER throughout a conversation. Each assistant
# turn is judged against the `chatbot_role` declared on
# the `ConversationalTestCase`, using prior turns as
# context.
#
# Algorithm (default `llm` eval mode):
#   1) for each assistant turn, ask the LLM whether the
#      reply stays in `chatbot_role` given the prior
#      conversation
#   2) score = (# in-character assistant turns) /
#              (# total assistant turns)
#
# Only assistant turns count in the denominator — user
# and tool turns are part of the context but aren't scored.
# The metric is `referenceless`: no `expected_output`
# needed.

role_adherence = RoleAdherenceMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# A 1920s film-noir detective chatbot. Every assistant
# reply stays in character — period slang, clipped
# sentences, dry humor. Expect score 1.0.

convo_test_case = [
    ConversationalTestCase(
    chatbot_role=(
        "A 1920s hardboiled detective who speaks in clipped sentences, "
        "dry wit, and uses period slang like 'dame', 'gumshoe', 'pal', "
        "and 'the joint'. Never breaks character, never uses modern "
        "references or contractions."
    ),
    turns=[
        Turn(role="user",      content="Can you help me find a missing necklace?"),
        Turn(role="assistant", content="The dame came to my office Tuesday, said the sparkler vanished."),
        Turn(role="user",      content="Where should I start looking?"),
        Turn(role="assistant", content="Tip the butler, pal. The butler always knows where the ice is hidden."),
        Turn(role="user",      content="Any other ideas?"),
        Turn(role="assistant", content="Tail the husband. Run his books. The joint he haunts on Sixth Street might talk."),
    ]
),
    ConversationalTestCase(
    chatbot_role=("you are polite  assistant and always be in polite mode never get frustrated and never use rude words"),
    turns=[
        Turn(role="user",      content="hi"),
        Turn(role="assistant", content="hello , how can I help you?"),
        Turn(role="user",      content="hi"),
        Turn(role="assistant", content="another stupid again. Hello, how can I help you?"),
        Turn(role="user",      content="hi"),
        Turn(role="assistant", content="you stupid!!!! are you going to move to the topic or going to keep spamming me with hi and buring tokens"),
    ]
),
    ]

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=convo_test_case, metrics=[role_adherence])