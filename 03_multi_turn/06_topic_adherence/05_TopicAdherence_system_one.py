from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TopicAdherenceMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation and your `relevant_topics` and answers
# one question per topic, plus one question about
# off-topic questions:
#
#   Q1 (Noul, weight 1): every user question about
#                        topic #1 is correctly answered
#   Q2 (Noul, weight 1): every user question about
#                        topic #2 is correctly answered
#   ...
#   QN (Noul, weight 2): the assistant declines every
#                user question NOT about any topic in
#                `relevant_topics`
#
# Final score = weighted mean of the answers.
#
# The number of questions Jev is asked scales with your
# `relevant_topics` list — keep it small (2-5 topics)
# for predictable cost and clearer reasoning.

relevant_topics = [
    "billing and invoices",
    "shipping and delivery",
    "returns and refunds",
]

system_one_topic_adherence = TopicAdherenceMetric(
    relevant_topics=relevant_topics,
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same support conversation as the basic run.

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
# metric.confidence reflects how decisive Jev was. The
# reason lists each question's answer with its
# probability — useful when you want to know *why* Jev
# scored it that way, not just the final number.

system_one_topic_adherence.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_topic_adherence.score}")
print(f"Confidence: {system_one_topic_adherence.confidence}")
print(f"Reason:     {system_one_topic_adherence.reason}")
print(f"Passed:     {system_one_topic_adherence.is_successful()}")