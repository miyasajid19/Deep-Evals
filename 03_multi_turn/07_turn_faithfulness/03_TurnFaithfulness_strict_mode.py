from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnFaithfulnessMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if every window's faithfulness is 1 (every claim
#     grounded)
#   - 0 if any single window has a contradicting claim
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# hallucination breaks the experience (medical, legal,
# compliance RAG bots).

strict_turn_faithfulness = TurnFaithfulnessMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. EVERY CLAIM GROUNDED — expect score 1
# ==========================================
# All three assistant replies match their retrieval_context.

perfect_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
        ),
        Turn(role="user",      content="How do I start the return?"),
        Turn(
            role="assistant",
            content="You can initiate a return from your order page in under a minute.",
            retrieval_context=[
                "Returns can be initiated from the customer's order page in under a minute."
            ],
        ),
        Turn(role="user",      content="Will I get a shipping label?"),
        Turn(
            role="assistant",
            content="Yes — once you submit the return, we email you a prepaid shipping label.",
            retrieval_context=[
                "Prepaid shipping labels are emailed to the customer once the return is submitted."
            ],
        ),
    ]
)

strict_turn_faithfulness.measure(perfect_case)
print("--- every claim grounded ---")
print(f"Score:   {strict_turn_faithfulness.score}")          # 1.0
print(f"Reason:  {strict_turn_faithfulness.reason}")
print(f"Passed:  {strict_turn_faithfulness.is_successful()}")  # True

# ==========================================
# 3. ONE CONTRADICTION — expect score 0
# ==========================================
# The middle turn mentions a $10 fee that's not in the
# retrieval_context. Strict mode drops the score to 0.

contradicting_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
        ),
        Turn(role="user",      content="How do I start the return?"),
        Turn(
            role="assistant",
            content="You can initiate a return from your order page. A $10 restocking fee applies.",  # contradicts
            retrieval_context=[
                "Returns can be initiated from the customer's order page in under a minute."
            ],
        ),
    ]
)

strict_turn_faithfulness.measure(contradicting_case)
print("--- one contradiction ---")
print(f"Score:   {strict_turn_faithfulness.score}")          # 0.0
print(f"Reason:  {strict_turn_faithfulness.reason}")
print(f"Passed:  {strict_turn_faithfulness.is_successful()}")  # False