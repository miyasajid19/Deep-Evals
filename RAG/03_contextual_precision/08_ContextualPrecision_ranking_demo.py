from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualPrecisionMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# Same retrieval contents, two different orderings.
# Contextual Precision's whole point is that order
# matters — re-running the same chunks in a different
# order should produce a different score. This script
# makes that property visible side-by-side.

contextual_precision = ContextualPrecisionMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. TWO RETRIEVAL CONTEXTS, SAME CHUNKS
# ==========================================
# Three chunks: one relevant, two unrelated.
# - GOOD_ORDER puts the relevant node on top.
# - BAD_ORDER buries the relevant node under noise.

relevant_chunk = "All customers are eligible for a 30 day full refund at no extra cost."
noise_chunk_1 = "Our company was founded in 1998 and is privately owned."
noise_chunk_2 = "We also sell socks, hats, and jackets in our online store."

# ==========================================
# 3. CASE A — good ranking
# ==========================================
# Verdicts: [yes, no, no]
# WCP at n=3:
#   k=1: 1/1 * 1 = 1.0
#   k=2: 1/2 * 0 = 0.0
#   k=3: 1/3 * 0 = 0.0
#   sum  = 1.0  →  divided by # relevant (1)  →  1.0

good_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=[relevant_chunk, noise_chunk_1, noise_chunk_2],
)

# ==========================================
# 4. CASE B — bad ranking
# ==========================================
# Verdicts: [no, no, yes]
# WCP at n=3:
#   k=1: 1/1 * 0 = 0.0
#   k=2: 1/2 * 0 = 0.0
#   k=3: 1/3 * 1 = 0.333
#   sum  = 0.333  →  divided by # relevant (1)  →  0.333

bad_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output="We offer a 30-day full refund at no extra cost.",
    expected_output="You are eligible for a 30 day full refund at no extra cost.",
    retrieval_context=[noise_chunk_1, noise_chunk_2, relevant_chunk],
)

# ==========================================
# 5. RUN BOTH METRICS ON THE SAME TEST CASES
# ==========================================
# Same chunks → completely different scores. This is
# the re-ranker signal Contextual Precision is designed
# to surface.

contextual_precision.measure(good_case)
print("--- good order (relevant first) ---")
print(f"Score:   {contextual_precision.score}")          # ~1.0
print(f"Reason:  {contextual_precision.reason}")
print(f"Passed:  {contextual_precision.is_successful()}")

contextual_precision.measure(bad_case)
print("--- bad order (relevant last) ---")
print(f"Score:   {contextual_precision.score}")          # ~0.333
print(f"Reason:  {contextual_precision.reason}")
print(f"Passed:  {contextual_precision.is_successful()}")