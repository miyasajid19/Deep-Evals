from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualPrecisionMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# Same retrieval chunks, two different orderings. The
# metric's whole point is that order matters — re-running
# the same chunks in a different order should produce a
# different score. This script makes that property
# visible side-by-side.

turn_contextual_precision = TurnContextualPrecisionMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED FIXTURES
# ==========================================
expected_outcome = (
    "The chatbot must explain store policies clearly, including "
    "refund eligibility, return process, and shipping timelines."
)

relevant_chunk = "All customers are eligible for a 30 day full refund at no extra cost."
noise_chunk_1  = "Our company was founded in 1998 and is privately owned."
noise_chunk_2  = "We also sell socks, hats, and jackets in our online store."

# ==========================================
# 3. CASE A — good ranking
# ==========================================
# Verdicts: [yes, no, no]
# WCP at n=3:
#   k=1: 1/1 * 1 = 1.0
#   k=2: 1/2 * 0 = 0.0
#   k=3: 1/3 * 0 = 0.0
#   sum  = 1.0  →  divided by # useful (1)  →  1.0

good_case = ConversationalTestCase(
    expected_outcome=expected_outcome,
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[relevant_chunk, noise_chunk_1, noise_chunk_2],
        ),
    ]
)

# ==========================================
# 4. CASE B — bad ranking
# ==========================================
# Verdicts: [no, no, yes]
# WCP at n=3:
#   k=1: 1/1 * 0 = 0.0
#   k=2: 1/2 * 0 = 0.0
#   k=3: 1/3 * 1 = 0.333
#   sum  = 0.333  →  divided by # useful (1)  →  0.333

bad_case = ConversationalTestCase(
    expected_outcome=expected_outcome,
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(
            role="assistant",
            content="We offer a 30-day full refund at no extra cost.",
            retrieval_context=[noise_chunk_1, noise_chunk_2, relevant_chunk],
        ),
    ]
)

# ==========================================
# 5. RUN THE METRIC ON BOTH CASES
# ==========================================
turn_contextual_precision.measure(good_case)
print("--- good order (useful first) ---")
print(f"Score:   {turn_contextual_precision.score:.3f}")          # ~1.0
print(f"Reason:  {turn_contextual_precision.reason}")
print(f"Passed:  {turn_contextual_precision.is_successful()}")

turn_contextual_precision.measure(bad_case)
print("--- bad order (useful last) ---")
print(f"Score:   {turn_contextual_precision.score:.3f}")          # ~0.333
print(f"Reason:  {turn_contextual_precision.reason}")
print(f"Passed:  {turn_contextual_precision.is_successful()}")