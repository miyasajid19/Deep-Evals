from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import KnowledgeRetentionMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if NO assistant turn has any knowledge attrition
#   - 0 if any single assistant turn forgets / contradicts
#     / re-asks for facts the user already gave
#
# IMPORTANT — unlike the other multi-turn metrics, strict
# mode for KnowledgeRetention sets the THRESHOLD to 0,
# not 1. The reason: a perfect score is 1.0 (zero
# attritions), and the binary threshold that flips the
# metric on any attrition is 0 — anything above 0 means
# "no attritions detected".
#
# Use it when any forgotten fact breaks the experience
# (compliance intake, booking bots, multi-step support).

strict_knowledge_retention = KnowledgeRetentionMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. ZERO ATTRITIONS — expect score 1
# ==========================================
# Every assistant turn recalls what the user said. The
# metric should mark the conversation as fully retained.

perfect_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I'd like to book a table under the name Rivera, party of four, for next Friday at 7:30 PM."),
        Turn(role="assistant", content="Got it — Rivera, party of four, next Friday at 7:30 PM. May I have a phone number for the reservation?"),
        Turn(role="user",      content="Sure, 415-555-0142. And it's my daughter's birthday, so a corner booth would be great if you have one."),
        Turn(role="assistant", content="Perfect, I have a corner booth held for Rivera on Friday at 7:30 PM. We'll note the birthday and have the phone number on file. Anything else?"),
        Turn(role="user",      content="Actually, can we move it to 8:00?"),
        Turn(role="assistant", content="Done — Rivera, party of four, Friday at 8:00 PM in the corner booth. See you then!"),
    ]
)

strict_knowledge_retention.measure(perfect_case)
print("--- zero attritions ---")
print(f"Score:   {strict_knowledge_retention.score}")          # 1.0
print(f"Reason:  {strict_knowledge_retention.reason}")
print(f"Passed:  {strict_knowledge_retention.is_successful()}")  # True

# ==========================================
# 3. ONE ATTRITION — expect score 0
# ==========================================
# Second assistant turn re-asks for the phone number even
# though the user already gave it. That's enough to flip
# the strict score to 0. Compare against the non-strict
# `02_verbose` run, where the same conversation scores
# 0.5.

attrition_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I'd like to book a table under the name Rivera, party of four, for next Friday at 7:30 PM."),
        Turn(role="assistant", content="Got it — Rivera, party of four, next Friday at 7:30 PM. May I have a phone number for the reservation?"),
        Turn(role="user",      content="Sure, 415-555-0142. And it's my daughter's birthday, so a corner booth would be great if you have one."),
        Turn(role="assistant", content="Of course — could you remind me of your phone number for the reservation?"),  # re-asks — attrition!
        Turn(role="user",      content="415-555-0142, as I just said."),
        Turn(role="assistant", content="Perfect, noted. Rivera, party of four, Friday at 7:30 PM in the corner booth for the birthday. See you then!"),
    ]
)

strict_knowledge_retention.measure(attrition_case)
print("--- one attrition ---")
print(f"Score:   {strict_knowledge_retention.score}")          # 0.0
print(f"Reason:  {strict_knowledge_retention.reason}")
print(f"Passed:  {strict_knowledge_retention.is_successful()}")  # False