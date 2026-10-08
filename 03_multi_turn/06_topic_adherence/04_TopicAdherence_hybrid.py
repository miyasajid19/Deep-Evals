from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TopicAdherenceMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Topic Adherence: the LLM STILL extracts the
# QA pairs from the conversation (this is the only LLM
# step that remains). For each pair, Jev (a System One
# model) classifies it with TWO yes/no questions:
#
#   Q1: is the user question on one of the
#       `relevant_topics`?
#   Q2: does the assistant actually answer it?
#
# Together the two answers map to the same TP / TN /
# FP / FN verdict as the LLM judge. The LLM-written
# reason is unchanged. If a Jev call fails, the metric
# silently defers that decision to the LLM.

relevant_topics = [
    "billing and invoices",
    "shipping and delivery",
    "returns and refunds",
]

hybrid_topic_adherence = TopicAdherenceMetric(
    relevant_topics=relevant_topics,
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still extracts QA pairs
    include_reason=True,
    eval_mode="hybrid",         # Jev classifies each pair
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same support conversation as the basic run — refund
# and shipping are on-topic (TP), Mars is declined (TN).

convo_test_case = ConversationalTestCase(
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
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reports how confident the system_one
# model was on average across its per-pair calls.

hybrid_topic_adherence.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_topic_adherence.score}")
print(f"Confidence: {hybrid_topic_adherence.confidence}")
print(f"Reason:     {hybrid_topic_adherence.reason}")
print(f"Passed:     {hybrid_topic_adherence.is_successful()}")