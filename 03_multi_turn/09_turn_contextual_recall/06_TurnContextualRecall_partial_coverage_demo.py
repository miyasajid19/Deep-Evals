from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRecallMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three conversations that share
# the same expected_outcome (refund eligibility + return
# process + shipping timelines) but differ in which facts
# the retrieval_context covers. Each missing fact drops
# one full numerator slot.

turn_contextual_recall = TurnContextualRecallMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED EXPECTED_OUTCOME
# ==========================================
expected_outcome = (
    "The chatbot must explain store policies clearly, including "
    "refund eligibility, return process, and shipping timelines."
)

# ==========================================
# 3. CASE A — full coverage
# ==========================================
# All three statements attributable. Expect score 1.0.

full_case = ConversationalTestCase(
    expected_outcome=expected_outcome,
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost. Returns can be initiated from your order page. Standard shipping arrives in 5-7 business days.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost.",
                "Returns can be initiated from the customer's order page in under a minute.",
                "Standard shipping arrives in 5-7 business days.",
            ],
        ),
    ]
)

# ==========================================
# 4. CASE B — one fact missing
# ==========================================
# "Shipping timelines" never gets retrieved. Expect 2/3 ≈ 0.667.

one_missing_case = ConversationalTestCase(
    expected_outcome=expected_outcome,
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

# ==========================================
# 5. CASE C — only refund is covered
# ==========================================
# Only the refund fact is present. Expect 1/3 ≈ 0.333.

only_refund_case = ConversationalTestCase(
    expected_outcome=expected_outcome,
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

# ==========================================
# 6. RUN THE METRIC ON ALL THREE CASES
# ==========================================
def run(label, case):
    turn_contextual_recall.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {turn_contextual_recall.score:.3f}")
    print(f"  Passed: {turn_contextual_recall.is_successful()}")
    print(f"  Reason: {turn_contextual_recall.reason}")

run("A — full coverage",     full_case)
run("B — one fact missing",  one_missing_case)
run("C — only refund covered", only_refund_case)