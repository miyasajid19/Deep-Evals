from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnFaithfulnessMetric

# ==========================================
# 1. THE METRICS (window_size = 2 vs. window_size = 10)
# ==========================================
# `window_size` controls how many recent turns are
# bundled per faithfulness judgment (default 10). Lower
# it when only recent context matters; raise it when
# older retrieval_context still counts.
#
# Here we run the SAME conversation twice — once with
# `window_size=2` (judge sees only the last 2 turns when
# scoring each assistant reply) and once with the default
# `window_size=10`.

window_small = TurnFaithfulnessMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    window_size=2,
)

window_default = TurnFaithfulnessMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    window_size=10,        # same as default
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Three turns. The first assistant reply is grounded in
# its own retrieval_context; the second leans on the
# retrieval_context of the PREVIOUS assistant turn (no
# retrieval_context of its own). With `window_size=2`,
# the judge sees both turns and the cross-context claim
# is fine. With `window_size=10` the answer doesn't
# change for a short conversation, but the score and
# per-window verdicts surface clearly.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost."
            ],
        ),
        Turn(role="user",      content="How long do refunds take?"),
        Turn(
            role="assistant",
            content="Refunds typically take 3-5 business days to land on your card.",
            retrieval_context=[
                # No retrieval_context of its own — claim leans on the
                # previous turn's retrieval_context via the window.
            ],
        ),
    ]
)

# ==========================================
# 3. RUN BOTH METRICS ON THE SAME TEST CASE
# ==========================================
window_small.measure(convo_test_case)
print("--- window_size=2 (recent turns only) ---")
print(f"Score:   {window_small.score}")
print(f"Reason:  {window_small.reason}")
print(f"Passed:  {window_small.is_successful()}")

window_default.measure(convo_test_case)
print("--- window_size=10 (full history) ---")
print(f"Score:   {window_default.score}")
print(f"Reason:  {window_default.reason}")
print(f"Passed:  {window_default.is_successful()}")