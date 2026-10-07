from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric

# ==========================================
# 1. THE METRIC (with advanced params)
# ==========================================
# Two knobs that aren't in the basic example:
#
#   truths_extraction_limit
#       Cap how many factual truths the LLM pulls from
#       `retrieval_context`. The LLM orders them by
#       importance. Useful when your retrieval returns
#       a long passage and you don't want it to anchor on
#       boilerplate (e.g. footer disclaimers).
#
#   penalize_ambiguous_claims
#       When True, claims the LLM judges as ambiguous
#       (neither clearly faithful nor clearly contradictory)
#       are NOT counted as faithful. Default False —
#       ambiguous claims count as faithful.

# --- Metric A: limited truths, default ambiguity handling ---
faithfulness_limited = FaithfulnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    truths_extraction_limit=2,         # only the top-2 facts from context
)

# --- Metric B: same, but penalize ambiguous claims too ---
faithfulness_strict_ambiguous = FaithfulnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    truths_extraction_limit=2,
    penalize_ambiguous_claims=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# A long retrieval_context where only the first two
# sentences carry the actual refund policy. The rest
# is boilerplate (return-shipping ETA, contact info).
# With `truths_extraction_limit=2`, the LLM should
# anchor on the policy sentences instead of the
# boilerplate.

retrieval_context = [
    "All customers are eligible for a 30 day full refund at no extra cost.",  # fact 1
    "Refunds can be initiated from the customer's order page in under a minute.",  # fact 2
    "Standard return shipping takes 5-7 business days.",                       # boilerplate
    "Contact our support team at support@example.com for help.",               # boilerplate
]

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "We offer a 30-day full refund at no extra cost. "
        "You can initiate a return from your order page."
    ),
    retrieval_context=retrieval_context,
)

# ==========================================
# 3. RUN BOTH METRICS ON THE SAME TEST CASE
# ==========================================
# Compare side-by-side:
#   - the default run tolerates ambiguity (claim counts
#     as faithful if the LLM isn't sure)
#   - the strict run punishes ambiguity (ambiguous claims
#     don't count as faithful, so the denominator shrinks)

faithfulness_limited.measure(test_case)
print("--- truths_extraction_limit=2 only ---")
print(f"Score:   {faithfulness_limited.score}")
print(f"Reason:  {faithfulness_limited.reason}")
print(f"Passed:  {faithfulness_limited.is_successful()}")

faithfulness_strict_ambiguous.measure(test_case)
print("--- + penalize_ambiguous_claims=True ---")
print(f"Score:   {faithfulness_strict_ambiguous.score}")
print(f"Reason:  {faithfulness_strict_ambiguous.reason}")
print(f"Passed:  {faithfulness_strict_ambiguous.is_successful()}")