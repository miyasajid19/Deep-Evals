from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

# ==========================================
# 1. THE METRICS
# ==========================================
# Contextual Precision and Contextual Recall are
# complementary views of the same retriever:
#
#   - Precision = "of the chunks I retrieved, are the
#                   relevant ones ranked above the
#                   irrelevant ones?"
#   - Recall    = "of the facts the ideal answer needs,
#                   how many did I actually retrieve?"
#
# This script runs both metrics on three retrieval
# patterns so you can see precision and recall move
# independently:
#
#   pattern A — perfect: relevant node ranked first,
#               ideal answer fully supported
#   pattern B — wrong order: relevant node present but
#               buried; ideal answer fully supported
#   pattern C — missing chunk: ideal answer needs a fact
#               that was never retrieved

precision = ContextualPrecisionMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)
recall = ContextualRecallMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED FIXTURES
# ==========================================
input_q = "What if these shoes don't fit?"
actual_output = "We offer a 30-day full refund at no extra cost."

# ideal answer needs TWO facts (refund policy + return page)
expected_output = (
    "You are eligible for a 30 day full refund at no extra cost. "
    "You can initiate a return from your order page."
)

relevant_chunk = "All customers are eligible for a 30 day full refund at no extra cost."
page_chunk    = "Returns can be initiated from the customer's order page in under a minute."
noise_chunk   = "Our company was founded in 1998 and is privately owned."

# ==========================================
# 3. PATTERN A — perfect ranking AND full coverage
# ==========================================
# Expect: high precision AND high recall.

case_a = LLMTestCase(
    input=input_q,
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=[relevant_chunk, page_chunk, noise_chunk],
)

# ==========================================
# 4. PATTERN B — wrong order, full coverage
# ==========================================
# The relevant chunks are present (recall = 1) but a
# noise chunk is ranked first (precision drops).
# Expect: low precision, high recall.

case_b = LLMTestCase(
    input=input_q,
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=[noise_chunk, relevant_chunk, page_chunk],
)

# ==========================================
# 5. PATTERN C — missing chunk
# ==========================================
# The "order page" fact was never retrieved, so the
# expected answer can't be fully supported.
# Expect: high precision (no noise) but low recall.

case_c = LLMTestCase(
    input=input_q,
    actual_output=actual_output,
    expected_output=expected_output,
    retrieval_context=[relevant_chunk],
)

# ==========================================
# 6. RUN BOTH METRICS ON ALL THREE CASES
# ==========================================
# Print P / R side by side for each pattern.

def run(label, case):
    precision.measure(case)
    recall.measure(case)
    print(f"--- {label} ---")
    print(f"  Precision: {precision.score:.3f}")
    print(f"  Recall:    {recall.score:.3f}")

run("A — perfect ranking, full coverage", case_a)
run("B — wrong order, full coverage",     case_b)
run("C — missing chunk, perfect order",   case_c)