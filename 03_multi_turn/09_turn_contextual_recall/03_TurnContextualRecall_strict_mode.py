from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRecallMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if every expected_outcome statement is
#     attributable to retrieval_context across the
#     conversation
#   - 0 if any single statement is unattributable
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# missing retrieval context breaks the experience
# (compliance intake, booking bots, multi-step support).

strict_turn_contextual_recall = TurnContextualRecallMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. FULLY ATTRIBUTABLE — expect score 1
# ==========================================
# Both expected statements trace back to nodes in the
# retrieval_context.

perfect_case = ConversationalTestCase(
    expected_outcome=(
        "The chatbot must explain store policies clearly, including "
        "refund eligibility and shipping timelines."
    ),
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost. Standard shipping arrives in 5-7 business days.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost.",
                "Standard shipping arrives in 5-7 business days.",
            ],
        ),
    ]
)

strict_turn_contextual_recall.measure(perfect_case)
print("--- fully attributable ---")
print(f"Score:   {strict_turn_contextual_recall.score}")          # 1.0
print(f"Reason:  {strict_turn_contextual_recall.reason}")
print(f"Passed:  {strict_turn_contextual_recall.is_successful()}")  # True

# ==========================================
# 3. PARTIAL COVERAGE — expect score 0
# ==========================================
# The "shipping timelines" statement is missing from the
# retrieval_context. Strict mode drops the score to 0.

partial_case = ConversationalTestCase(
    expected_outcome=(
        "The chatbot must explain store policies clearly, including "
        "refund eligibility and shipping timelines."
    ),
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost.",
            ],
        ),
    ]
)

strict_turn_contextual_recall.measure(partial_case)
print("--- partial coverage ---")
print(f"Score:   {strict_turn_contextual_recall.score}")          # 0.0
print(f"Reason:  {strict_turn_contextual_recall.reason}")
print(f"Passed:  {strict_turn_contextual_recall.is_successful()}")  # False