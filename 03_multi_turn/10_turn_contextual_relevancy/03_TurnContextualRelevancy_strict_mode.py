from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRelevancyMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if every retrieval_context statement is relevant
#   - 0 if any single statement is irrelevant
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# noisy chunk breaks the experience (compliance, legal,
# tightly-scoped RAG bots).

strict_turn_contextual_relevancy = TurnContextualRelevancyMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. FULLY RELEVANT — expect score 1
# ==========================================
# Every statement in the retrieval_context addresses the
# user's question.

perfect_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost. Returns can be initiated from your order page.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost.",
                "Returns can be initiated from the customer's order page in under a minute.",
            ],
        ),
    ]
)

strict_turn_contextual_relevancy.measure(perfect_case)
print("--- fully relevant ---")
print(f"Score:   {strict_turn_contextual_relevancy.score}")          # 1.0
print(f"Reason:  {strict_turn_contextual_relevancy.reason}")
print(f"Passed:  {strict_turn_contextual_relevancy.is_successful()}")  # True

# ==========================================
# 3. ONE NOISE STATEMENT — expect score 0
# ==========================================
# The retrieval_context contains a noise statement
# ("founded in 1998"). Strict mode drops the score to 0.

noisy_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost. Our company was founded in 1998.",
            ],
        ),
    ]
)

strict_turn_contextual_relevancy.measure(noisy_case)
print("--- one noise statement ---")
print(f"Score:   {strict_turn_contextual_relevancy.score}")          # 0.0
print(f"Reason:  {strict_turn_contextual_relevancy.reason}")
print(f"Passed:  {strict_turn_contextual_relevancy.is_successful()}")  # False