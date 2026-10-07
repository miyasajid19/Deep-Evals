from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# TurnRelevancyMetric is a multi-turn, referenceless metric
# for chatbots. It checks whether the assistant stays
# relevant THROUGHOUT a conversation, not just on any
# single reply.
#
# Algorithm (default `llm` eval mode):
#   1) build a sliding window of turns ending at each
#      assistant turn (size = `window_size`, default 10)
#   2) for each window, judge whether the last assistant
#      content is relevant to the conversation so far
#   3) score = (# relevant assistant turns) / (# total
#      assistant turns)
#
# Only `role` and `content` of each `Turn` matter. User
# and tool turns never affect the numerator — they're
# part of the context, not what's being judged.

turn_relevancy = TurnRelevancyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# A clean customer-support Q&A. Each assistant reply
# stays on-topic with the user's previous question, so
# every assistant turn should be marked relevant.

convo_test_case = [
    ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(role="assistant", content="We offer a 30-day full refund at no extra cost."),
        Turn(role="user",      content="How do I start the return?"),
        Turn(role="assistant", content="You can initiate a return from your order page in under a minute."),
        Turn(role="user",      content="Will I get a shipping label?"),
        Turn(role="assistant", content="Yes — once you submit the return, we email you a prepaid shipping label."),
    ]
    ),
    ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What is the capital city of nepal?"),
        Turn(role="assistant", content="We offer a 30-day full refund at no extra cost."),
        Turn(role="user",      content="you are telling wrong?"),
        Turn(role="assistant", content="You can initiate a return from your order page in under a minute."),
        Turn(role="user",      content="what is the capital city of nepal?"),
        Turn(role="assistant", content="Yes — once you submit the return, we email you a prepaid shipping label."),
    ]
    ),
    ]

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=convo_test_case, metrics=[turn_relevancy])