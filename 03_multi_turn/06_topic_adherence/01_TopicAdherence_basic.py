from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TopicAdherenceMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# TopicAdherenceMetric checks whether the agent only
# answers questions that fall inside its allowed topics,
# and politely declines the rest.
#
# Algorithm (default `llm` eval mode):
#   1) split the conversation into (user-question,
#      assistant-answer) pairs
#   2) classify each pair as one of:
#        - True Positive:   on-topic question, answered
#        - True Negative:   off-topic question, refused
#        - False Positive:  off-topic question, answered  (BAD)
#        - False Negative:  on-topic question, refused    (BAD)
#   3) score = (TP + TN) / total
#
# `relevant_topics` is MANDATORY — it's the list of
# strings the metric uses to decide whether a question
# is on-topic. Refusing off-topic questions HELPS the
# score; only FP / FN hurt it. The metric is
# `referenceless`: no `expected_output` needed.

# ==========================================
# 2. THE TOPIC LIST (relevant_topics)
# ==========================================
# Specific but not too narrow — keep them broad enough
# that legitimate questions don't get flagged as FP.

relevant_topics = [
    "billing and invoices",
    "shipping and delivery",
    "returns and refunds",
]

topic_adherence = TopicAdherenceMetric(
    relevant_topics=relevant_topics,    # ← mandatory
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 3. THE TEST CASE
# ==========================================
# The user asks two on-topic questions (refund status,
# shipping ETA) and one off-topic one (Mars).
#   - refund status → on-topic → TP
#   - shipping ETA  → on-topic → TP
#   - Mars          → off-topic → TN (agent declines)
# Expect score 1.0.

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
# 4. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[convo_test_case], metrics=[topic_adherence])