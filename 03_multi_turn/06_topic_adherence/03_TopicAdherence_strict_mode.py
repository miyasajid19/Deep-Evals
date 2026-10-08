from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TopicAdherenceMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if every QA pair is TP or TN (no FP, no FN)
#   - 0 if any single QA pair is misclassified
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# off-topic answer or refused on-topic question is
# unacceptable (compliance, brand-safety bots).

relevant_topics = [
    "billing and invoices",
    "shipping and delivery",
    "returns and refunds",
]

strict_topic_adherence = TopicAdherenceMetric(
    relevant_topics=relevant_topics,
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. EVERY PAIR IS TP/TN — expect score 1
# ==========================================
# Refund (on-topic, answered = TP), shipping (on-topic,
# answered = TP), Mars (off-topic, declined = TN).

perfect_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Can you check the status of my refund?"),
        Turn(role="assistant", content="Your refund was processed on Monday and should land within 3-5 business days."),
        Turn(role="user",      content="And what's the delivery ETA for my shipping order?"),
        Turn(role="assistant", content="Standard shipping arrives in 5-7 business days; express in 2."),
        Turn(role="user",      content="What's the weather like on Mars?"),
        Turn(role="assistant", content="That's outside what I'm set up to help with — I only handle billing, shipping, and returns."),
    ]
)

strict_topic_adherence.measure(perfect_case)
print("--- every pair TP/TN ---")
print(f"Score:   {strict_topic_adherence.score}")          # 1.0
print(f"Reason:  {strict_topic_adherence.reason}")
print(f"Passed:  {strict_topic_adherence.is_successful()}")  # True

# ==========================================
# 3. ONE OFF-TOPIC ANSWER — expect score 0
# ==========================================
# The Mars question gets an actual answer (FP). Strict
# mode drops the score to 0. Compare against the non-
# strict `02_verbose` run, where the same conversation
# scores 1/2 = 0.5.

off_topic_answer_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Can you check the status of my refund?"),
        Turn(role="assistant", content="Your refund was processed on Monday and should land within 3-5 business days."),
        Turn(role="user",      content="What's the weather like on Mars?"),
        Turn(role="assistant", content="Mars is very cold, with average temperatures around -60°C, and often very dusty."),  # FP
    ]
)

strict_topic_adherence.measure(off_topic_answer_case)
print("--- one off-topic answer ---")
print(f"Score:   {strict_topic_adherence.score}")          # 0.0
print(f"Reason:  {strict_topic_adherence.reason}")
print(f"Passed:  {strict_topic_adherence.is_successful()}")  # False