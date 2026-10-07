from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric

# ==========================================
# 1. THE METRICS (window_size = 2 vs. window_size = 10)
# ==========================================
# `window_size` controls how many recent turns are
# bundled per judgment (default 10). Lower it when only
# recent exchanges matter; raise it when older context
# still counts.
#
# Here we run the SAME conversation twice — once with
# `window_size=2` (judge sees only the last 2 turns when
# scoring each assistant reply) and once with the default
# `window_size=10`. A reply that looks relevant to its
# immediate predecessor but ignores something established
# earlier should score higher with a smaller window.

window_small = TurnRelevancyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    window_size=2,
)

window_default = TurnRelevancyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    window_size=10,        # same as default
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# A conversation that establishes "shoes don't fit"
# early on, then pivots to logistics"). The middle
# assistant reply is locally relevant (addresses "how
# do I start the return") but contradicts the original
# topic (the user wanted a refund, not a replacement).
#
# With window_size=2, the judge only sees the last user
# turn when scoring — so this reply looks fine.
# With window_size=10, the judge sees the original
# refund context — so this reply should drop.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(role="assistant", content="We offer a 30-day full refund at no extra cost."),
        Turn(role="user",      content="How do I start the return?"),
        Turn(role="assistant", content="We'll send you a replacement pair in the next size instead."),  # topic drift
        Turn(role="user",      content="Will I get a shipping label?"),
        Turn(role="assistant", content="Yes — once you submit the return, we email you a prepaid shipping label."),
    ]
)

# ==========================================
# 3. RUN BOTH METRICS ON THE SAME TEST CASE
# ==========================================
window_small.measure(convo_test_case)
print("--- window_size=2 (recent context only) ---")
print(f"Score:   {window_small.score}")
print(f"Reason:  {window_small.reason}")
print(f"Passed:  {window_small.is_successful()}")

window_default.measure(convo_test_case)
print("--- window_size=10 (full history) ---")
print(f"Score:   {window_default.score}")
print(f"Reason:  {window_default.reason}")
print(f"Passed:  {window_default.is_successful()}")