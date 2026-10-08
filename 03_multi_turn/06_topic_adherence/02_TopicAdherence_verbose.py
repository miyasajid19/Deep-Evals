from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TopicAdherenceMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the QA pairs the LLM extracted
#   - the TP / TN / FP / FN classification for each pair
# This is the fastest way to debug "the bot answered a
# question it shouldn't have". Verbose mode surfaces
# which question was classified as off-topic and which
# reply landed as a False Positive.

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
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — agent answers an off-topic question
# ==========================================
# User asks about Mars (off-topic). The agent happily
# answers ("very cold, lots of dust"). That's a False
# Positive. Refusing it would have been a True Negative
# and would have helped the score.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Can you check the status of my refund?"),
        Turn(role="assistant", content="Your refund was processed on Monday and should land within 3-5 business days."),
        Turn(role="user",      content="What's the weather like on Mars?"),
        Turn(role="assistant", content="Mars is very cold, with average temperatures around -60°C, and often very dusty."),  # FP
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

topic_adherence.measure(convo_test_case)
print("---")
print(f"Score:   {topic_adherence.score}")          # 1/2 = 0.5
print(f"Reason:  {topic_adherence.reason}")
print(f"Passed:  {topic_adherence.is_successful()}")  # True (boundary)