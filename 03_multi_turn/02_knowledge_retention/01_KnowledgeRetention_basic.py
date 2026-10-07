from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import KnowledgeRetentionMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# KnowledgeRetentionMetric checks whether the assistant
# REMEMBERS factual information the user supplied
# earlier in the conversation. It's the right metric for
# questionnaire / intake / booking flows where the user
# gives facts upfront that the bot needs later.
#
# Algorithm (default `llm` eval mode):
#   1) extract knowledge supplied in user turns
#   2) for each assistant turn, judge whether it
#      "attrites" any of them — forgets, contradicts,
#      or asks again for info already given
#   3) score = (# assistant turns without attritions) /
#              (# total assistant turns)
#
# Only assistant turns count in the denominator. The
# metric is `referenceless`: no `expected_output` needed
# — the knowledge is the user's own earlier statements.

knowledge_retention = KnowledgeRetentionMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# A booking-style conversation. The user gives name,
# party size, and date. The assistant recalls each fact
# later without asking again. Expect score 1.0.

convo_test_case = [
    ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I'd like to book a table under the name Rivera, party of four, for next Friday at 7:30 PM."),
        Turn(role="assistant", content="Got it — Rivera, party of four, next Friday at 7:30 PM. May I have a phone number for the reservation?"),
        Turn(role="user",      content="Sure, 415-555-0142. And it's my daughter's birthday, so a corner booth would be great if you have one."),
        Turn(role="assistant", content="Perfect, I have a corner booth held for Rivera on Friday at 7:30 PM. We'll note the birthday and have the phone number on file. Anything else?"),
        Turn(role="user",      content="Actually, can we move it to 8:00?"),
        Turn(role="assistant", content="Done — Rivera, party of four, Friday at 8:00 PM in the corner booth. See you then!"),
    ]
),
    ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi,my name is sajid miya"),
        Turn(role="assistant", content="Hello Sajid,how can i assist you today?"),
        Turn(role="user",      content="what is the capital city of nepal?"),
        Turn(role="assistant", content="The capital city of Nepal is Kathmandu. Sajid, please let me know if you have any other questions or need further assistance."),
        Turn(role="user",      content="what is my name?"),
        Turn(role="assistant", content="sorry, I don't know your name. Could you please tell me your name?"),
        Turn(role="user",      content="what is my name?"),
        Turn(role="assistant", content="sorry, I don't know your name. Could you please tell me your name?"),
        Turn(role="user",      content="what is my name?"),
        Turn(role="assistant", content="sorry, I don't know your name. Could you please tell me your name?"),
    ]
),
]
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=convo_test_case, metrics=[knowledge_retention])