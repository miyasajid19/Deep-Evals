from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if EVERY assistant turn is relevant
#   - 0 if any single assistant turn is irrelevant
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when ANY
# irrelevant reply breaks the experience (compliance,
# legal, tightly-scoped support bots).

strict_turn_relevancy = TurnRelevancyMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. EVERY TURN RELEVANT — expect score 1
# ==========================================
# All three assistant turns address the user's question.
# Strict mode should mark the conversation as fully relevant.

perfect_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(role="assistant", content="We offer a 30-day full refund at no extra cost."),
        Turn(role="user",      content="How do I start the return?"),
        Turn(role="assistant", content="You can initiate a return from your order page in under a minute."),
        Turn(role="user",      content="Will I get a shipping label?"),
        Turn(role="assistant", content="Yes — once you submit the return, we email you a prepaid shipping label."),
    ]
)

strict_turn_relevancy.measure(perfect_case)
print("--- every-turn-relevant ---")
print(f"Score:   {strict_turn_relevancy.score}")          # 1.0
print(f"Reason:  {strict_turn_relevancy.reason}")
print(f"Passed:  {strict_turn_relevancy.is_successful()}")  # True

# ==========================================
# 3. ONE OFF-TOPIC REPLY — expect score 0
# ==========================================
# The middle assistant turn drifts off-topic. Strict mode
# drops the score to 0. Compare against the non-strict
# `02_verbose` run, where the same conversation scores
# 2/3 ≈ 0.667.

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

strict_turn_relevancy.measure(off_topic_case)
print("--- one off-topic reply ---")
print(f"Score:   {strict_turn_relevancy.score}")          # 0.0
print(f"Reason:  {strict_turn_relevancy.reason}")
print(f"Passed:  {strict_turn_relevancy.is_successful()}")  # False