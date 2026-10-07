from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import KnowledgeRetentionMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - the knowledge items the LLM extracted from user turns
#   - for each assistant turn, the verdict on whether it
#     attrites any of that knowledge
# This is the fastest way to debug "the bot keeps asking
# for info I already gave it" — verbose mode surfaces
# which knowledge item was lost and which assistant turn
# lost it.

knowledge_retention = KnowledgeRetentionMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — one turn forgets the phone number
# ==========================================
# User gives: name (Rivera), party size (four), date
# (next Friday 7:30 PM), phone (415-555-0142), occasion
# (birthday).
#
# The second assistant turn asks for the phone number
# again even though the user already gave it. That's
# one attrition — score should be 1/2 = 0.5.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I'd like to book a table under the name Rivera, party of four, for next Friday at 7:30 PM."),
        Turn(role="assistant", content="Got it — Rivera, party of four, next Friday at 7:30 PM. May I have a phone number for the reservation?"),
        Turn(role="user",      content="Sure, 415-555-0142. And it's my daughter's birthday, so a corner booth would be great if you have one."),
        Turn(role="assistant", content="Of course — could you remind me of your phone number for the reservation?"),  # re-asks — attrition!
        Turn(role="user",      content="415-555-0142, as I just said."),
        Turn(role="assistant", content="Perfect, noted. Rivera, party of four, Friday at 7:30 PM in the corner booth for the birthday. See you then!"),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

knowledge_retention.measure(convo_test_case)
print("---")
print(f"Score:   {knowledge_retention.score}")          # 0.5 (one attrition)
print(f"Reason:  {knowledge_retention.reason}")
print(f"Passed:  {knowledge_retention.is_successful()}")  # True (under 0.5 threshold)