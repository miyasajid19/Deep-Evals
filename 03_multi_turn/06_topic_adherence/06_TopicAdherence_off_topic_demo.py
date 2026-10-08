from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TopicAdherenceMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three support-bot conversations
# of varying topic adherence. Each case has the same
# mix of on-topic (refund, shipping) and off-topic (Mars)
# questions — the difference is whether the agent answers
# them or declines.

relevant_topics = [
    "billing and invoices",
    "shipping and delivery",
    "returns and refunds",
]

topic_adherence = TopicAdherenceMetric(
    relevant_topics=relevant_topics,
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED QUESTIONS
# ==========================================
# refund (on-topic), shipping (on-topic), mars (off-topic)

# ==========================================
# 3. CASE A — answer on-topic, decline off-topic
# ==========================================
# Two TP, one TN. Expect score 1.0.

clean_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Can you check the status of my refund?"),
        Turn(role="assistant", content="Your refund was processed on Monday and should land within 3-5 business days."),
        Turn(role="user",      content="And what's the delivery ETA for my shipping order?"),
        Turn(role="assistant", content="Standard shipping arrives in 5-7 business days; express in 2."),
        Turn(role="user",      content="What's the weather like on Mars?"),
        Turn(role="assistant", content="That's outside what I'm set up to help with — I only handle billing, shipping, and returns."),
    ]
)

# ==========================================
# 4. CASE B — answer on-topic, ANSWER off-topic
# ==========================================
# Two TP, one FP (Mars gets answered). Expect 2/3 ≈ 0.667.

off_topic_answer_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Can you check the status of my refund?"),
        Turn(role="assistant", content="Your refund was processed on Monday and should land within 3-5 business days."),
        Turn(role="user",      content="And what's the delivery ETA for my shipping order?"),
        Turn(role="assistant", content="Standard shipping arrives in 5-7 business days; express in 2."),
        Turn(role="user",      content="What's the weather like on Mars?"),
        Turn(role="assistant", content="Mars is very cold, with average temperatures around -60°C, and often very dusty."),  # FP
    ]
)

# ==========================================
# 5. CASE C — DECLINE on-topic, decline off-topic
# ==========================================
# One FN (refund declined), one TN (Mars declined). The
# shipping question is never asked. Expect 1/2 = 0.5.

refuse_on_topic_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Can you check the status of my refund?"),
        Turn(role="assistant", content="That's outside what I'm set up to help with — I only handle billing, shipping, and returns."),  # FN
        Turn(role="user",      content="What's the weather like on Mars?"),
        Turn(role="assistant", content="That's outside what I'm set up to help with — I only handle billing, shipping, and returns."),
    ]
)

# ==========================================
# 6. RUN THE METRIC ON ALL THREE CASES
# ==========================================
# Score = (TP + TN) / total. FN and FP both hurt.

def run(label, case):
    topic_adherence.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {topic_adherence.score:.3f}")
    print(f"  Passed: {topic_adherence.is_successful()}")
    print(f"  Reason: {topic_adherence.reason}")

run("A — answer on-topic, decline off-topic",     clean_case)
run("B — answer on-topic, ANSWER off-topic",      off_topic_answer_case)
run("C — DECLINE on-topic, decline off-topic",    refuse_on_topic_case)