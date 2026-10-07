from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnRelevancyMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - each sliding window the metric constructed
#   - the verdict (relevant / not) for the last assistant
#     turn in each window
# This is the fastest way to debug a low score. If an
# assistant reply looks fine on its own but the metric
# marks it irrelevant, the verbose log will show which
# earlier turn the judge anchored on.

turn_relevancy = TurnRelevancyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — middle reply drifts off-topic
# ==========================================
# Three assistant turns. The first and third are clearly
# relevant to the conversation; the middle one ignores
# the user's question and talks about the weather. Verbose
# mode will surface the per-window verdict.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="What if these shoes don't fit?"),
        Turn(role="assistant", content="We offer a 30-day full refund at no extra cost."),       # relevant
        Turn(role="user",      content="How do I start the return?"),
        Turn(role="assistant", content="It's sunny in Paris today, with a high of 24°C."),       # off-topic
        Turn(role="user",      content="Will I get a shipping label?"),
        Turn(role="assistant", content="Yes — once you submit the return, we email you a prepaid shipping label."),  # relevant
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

turn_relevancy.measure(convo_test_case)
print("---")
print(f"Score:   {turn_relevancy.score}")          # 2/3 ≈ 0.667
print(f"Reason:  {turn_relevancy.reason}")
print(f"Passed:  {turn_relevancy.is_successful()}")  # True (under 0.5 threshold it passes)