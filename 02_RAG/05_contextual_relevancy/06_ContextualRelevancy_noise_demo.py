from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ContextualRelevancyMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three retrieval setups that
# share the same user input and assistant reply but
# differ in how much of the retrieval_context is on-topic.
# The metric grades the RETRIEVER — irrelevant nodes
# lower the score even when the answer is correct.

contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED INPUT + REPLY
# ==========================================
input_q = "What if these shoes don't fit?"
actual_output = "We offer a 30-day full refund at no extra cost."

# ==========================================
# 3. CASE A — clean retrieval
# ==========================================
# Single relevant statement. Expect score 1.0.

clean_case = LLMTestCase(
    input=input_q,
    actual_output=actual_output,
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost.",
    ],
)

# ==========================================
# 4. CASE B — half noise
# ==========================================
# Two statements: one relevant, one noise. Expect 0.5.

half_noise_case = LLMTestCase(
    input=input_q,
    actual_output=actual_output,
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost. Our company was founded in 1998.",
    ],
)

# ==========================================
# 5. CASE C — mostly noise
# ==========================================
# Three statements: one relevant, two noise. Expect 1/3 ≈ 0.333.

mostly_noise_case = LLMTestCase(
    input=input_q,
    actual_output=actual_output,
    retrieval_context=[
        "All customers are eligible for a 30 day full refund at no extra cost. "
        "Our company was founded in 1998. "
        "We also sell socks, hats, and jackets."
    ],
)

# ==========================================
# 6. RUN THE METRIC ON ALL THREE CASES
# ==========================================
def run(label, case):
    contextual_relevancy.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {contextual_relevancy.score:.3f}")
    print(f"  Passed: {contextual_relevancy.is_successful()}")
    print(f"  Reason: {contextual_relevancy.reason}")

run("A — clean retrieval",  clean_case)
run("B — half noise",       half_noise_case)
run("C — mostly noise",     mostly_noise_case)