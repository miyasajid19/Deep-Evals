from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualPrecisionMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if every window's WCP is 1 (every useful node
#     ranked above every useless node)
#   - 0 otherwise
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# mis-ranked chunk is unacceptable (compliance, legal,
# tightly-scoped RAG bots).

strict_turn_contextual_precision = TurnContextualPrecisionMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. PERFECTLY RANKED — expect score 1
# ==========================================
# All useful nodes (1 and 2) are ranked above the
# unrelated one (3).

perfect_case = ConversationalTestCase(
    expected_outcome=(
        "The chatbot must explain store policies clearly, including "
        "refund eligibility, return process, and shipping timelines."
    ),
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost.",
                "Returns can be initiated from the customer's order page in under a minute.",
                "Our company was founded in 1998 and is privately owned.",
            ],
        ),
    ]
)

strict_turn_contextual_precision.measure(perfect_case)
print("--- perfect ranking ---")
print(f"Score:   {strict_turn_contextual_precision.score}")          # 1.0
print(f"Reason:  {strict_turn_contextual_precision.reason}")
print(f"Passed:  {strict_turn_contextual_precision.is_successful()}")  # True

# ==========================================
# 3. MIS-RANKED — expect score 0
# ==========================================
# The relevant node is BURIED below an unrelated one.
# WCP drops below 1. Strict mode rounds to 0.

misranked_case = ConversationalTestCase(
    expected_outcome=(
        "The chatbot must explain store policies clearly, including "
        "refund eligibility, return process, and shipping timelines."
    ),
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "Our company was founded in 1998 and is privately owned.",                   # noise ranked #1
                "All customers are eligible for a 30 day full refund at no extra cost.",    # relevant ranked #2
            ],
        ),
    ]
)

strict_turn_contextual_precision.measure(misranked_case)
print("--- mis-ranked ---")
print(f"Score:   {strict_turn_contextual_precision.score}")          # 0.0
print(f"Reason:  {strict_turn_contextual_precision.reason}")
print(f"Passed:  {strict_turn_contextual_precision.is_successful()}")  # False