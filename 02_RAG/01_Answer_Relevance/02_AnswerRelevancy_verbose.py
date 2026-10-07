from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate
# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the list of statements the LLM extracted from `actual_output`
#   - the verdict (relevant / not relevant) for each statement
# This is the fastest way to debug a low score — if the LLM
# split a single idea into many statements, or flagged an
# on-topic one as off-topic, you'll see it here.

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — answer rambles off-topic
# ==========================================
# The first sentence addresses the question, the rest is filler.
# Verbose mode will show which statements were marked relevant
# and which dragged the score down.

test_case = [
    LLMTestCase(
    input="What if these shoes don't fit?",
    actual_output=(
        "We offer a 30-day full refund at no extra cost. "
        "Our company was founded in 1998. "
        "We also sell socks, hats, and jackets. "
        "Please contact our support team to start a return."
    ),
    ),
    LLMTestCase(
    input="what is the capital city of nepal?",
    actual_output=("The capital city of Nepal is Kathmandu. "),
    ),
             ]

evaluate(test_cases=test_case, metrics=[answer_relevancy])