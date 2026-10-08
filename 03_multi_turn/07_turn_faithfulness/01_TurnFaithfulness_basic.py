from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnFaithfulnessMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# TurnFaithfulnessMetric evaluates whether the assistant
# is GROUNDED in the `retrieval_context` available at each
# turn. It's the multi-turn sibling of the single-turn
# `FaithfulnessMetric` — instead of one (input,
# actual_output, retrieval_context) tuple, you supply a
# `ConversationalTestCase` where each Turn carries its
# own retrieval_context.
#
# Algorithm (default `llm` eval mode):
#   1) build sliding windows of turns (size = `window_size`,
#      default 10)
#   2) for each window:
#        - extract truths from the retrieval_context(s) in
#          the window
#        - extract claims from each assistant turn
#        - judge each claim faithful vs contradictory
#   3) score = mean(per-window faithfulness)
#
# A claim is faithful if it doesn't CONTRADICT any fact
# in the window's retrieval_context — mere absence of
# evidence doesn't fail a claim, but contradiction does.
# The metric is `referenceBased` because the contexts
# are the reference; no `expected_output` is needed.

turn_faithfulness = TurnFaithfulnessMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# A RAG Q&A about refunds. Each assistant turn attaches
# the retrieved chunks it was grounded in. Every claim
# matches the relevant retrieval_context, so the score
# should be 1.0.

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
            content="You can initiate a return from your order page in under a minute.",
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
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and writes
# a report (Confident AI if configured, else local cache).

evaluate(test_cases=[convo_test_case], metrics=[turn_faithfulness])