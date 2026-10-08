from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnFaithfulnessMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - each sliding window the metric constructed
#   - the truths extracted from the window's context
#   - the claims extracted from the assistant's reply
#   - the verdict (faithful / contradicts) for each claim
# This is the fastest way to debug a low score. If the
# LLM hallucinated a fact the retrieval_context never
# supported, verbose mode surfaces which claim got
# flagged.

turn_faithfulness = TurnFaithfulnessMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — one turn contradicts its context
# ==========================================
# Turn 1 and 3 are grounded. Turn 2's reply mentions a
# restocking fee that the retrieval_context never said.
# Verbose mode will flag that specific claim.

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
        Turn(role="user",      content="How do I start the return?"),
        Turn(
            role="assistant",
            content="You can initiate a return from your order page. A $10 restocking fee applies.",  # contradicted by context
            retrieval_context=[
                "Returns can be initiated from the customer's order page in under a minute."
            ],
        ),
        Turn(role="user",      content="Will I get a shipping label?"),
        Turn(
            role="assistant",
            content="Yes — once you submit the return, we email you a prepaid shipping label.",
            retrieval_context=[
                "Prepaid shipping labels are emailed to the customer once the return is submitted."
            ],
        ),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

turn_faithfulness.measure(convo_test_case)
print("---")
print(f"Score:   {turn_faithfulness.score}")
print(f"Reason:  {turn_faithfulness.reason}")
print(f"Passed:  {turn_faithfulness.is_successful()}")