from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualPrecisionMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation — each turn carrying its `role`,
# `content`, and ordered `retrieval_context` — plus the
# `expected_outcome`, and answers three questions about
# the ORDERING:
#
#   Q1 (Noul, weight 2): in every assistant turn, useful
#                        documents are listed before not
#                        useful ones
#   Q2 (Noul, weight 1): in every assistant turn, the
#                        first document is useful
#   Q3 (Score, weight 1): how well the documents are
#                        ordered across assistant turns
#                        (all-useful-at-end → all-useful-first)
#
# Final score = weighted mean of the three answers.

system_one_turn_contextual_precision = TurnContextualPrecisionMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same setup as the basic run.

convo_test_case = ConversationalTestCase(
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
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
        ),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reflects how decisive Jev was. The
# reason lists each question's answer with its
# probability — useful when you want to know *why* Jev
# scored it that way, not just the final number.

system_one_turn_contextual_precision.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_turn_contextual_precision.score}")
print(f"Confidence: {system_one_turn_contextual_precision.confidence}")
print(f"Reason:     {system_one_turn_contextual_precision.reason}")
print(f"Passed:     {system_one_turn_contextual_precision.is_successful()}")