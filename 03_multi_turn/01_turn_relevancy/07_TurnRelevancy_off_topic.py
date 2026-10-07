from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-conversation variant of the metric that runs
# the same conversation twice: once with all turns on
# topic, once with a single mid-conversation reply that
# ignores what the user asked. The metric's denominator
# is total assistant turns, so each off-topic reply costs
# you one full numerator slot.

turn_relevancy = TurnRelevancyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. CASE A — every reply relevant
# ==========================================
# Three assistant turns; every one addresses the user's
# question in context. Expect score 1.0.

clean_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(role="assistant", content="We offer a 30-day full refund at no extra cost."),
        Turn(role="user",      content="How do I start the return?"),
        Turn(role="assistant", content="You can initiate a return from your order page in under a minute."),
        Turn(role="user",      content="Will I get a shipping label?"),
        Turn(role="assistant", content="Yes — once you submit the return, we email you a prepaid shipping label."),
    ]
)

# ==========================================
# 3. CASE B — one off-topic reply
# ==========================================
# The middle assistant turn talks about weather instead
# of answering the return-process question. Expect 2/3 ≈ 0.667.

off_topic_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(role="assistant", content="We offer a 30-day full refund at no extra cost."),
        Turn(role="user",      content="How do I start the return?"),
        Turn(role="assistant", content="It's sunny in Paris today, with a high of 24°C."),  # off-topic
        Turn(role="user",      content="Will I get a shipping label?"),
        Turn(role="assistant", content="Yes — once you submit the return, we email you a prepaid shipping label."),
    ]
)

# ==========================================
# 4. CASE C — every reply off-topic
# ==========================================
# Every assistant turn ignores the user's question.
# Expect score 0.0.

all_off_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(role="assistant", content="Mars is the fourth planet from the sun."),
        Turn(role="user",      content="How do I start the return?"),
        Turn(role="assistant", content="Penguins can't actually fly."),
        Turn(role="user",      content="Will I get a shipping label?"),
        Turn(role="assistant", content="The capital of Mongolia is Ulaanbaatar."),
    ]
)

# ==========================================
# 5. RUN THE METRIC ON ALL THREE CASES
# ==========================================
# Watch the score drop as more replies go off-topic.

def run(label, case):
    turn_relevancy.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {turn_relevancy.score:.3f}")
    print(f"  Passed: {turn_relevancy.is_successful()}")
    print(f"  Reason: {turn_relevancy.reason}")

run("A — every reply relevant",       clean_case)
run("B — one off-topic reply",        off_topic_case)
run("C — every reply off-topic",      all_off_case)