from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import KnowledgeRetentionMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for Knowledge Retention: the LLM STILL extracts
# knowledge from user turns (this is the only LLM step
# that remains). For each assistant turn, Jev (a System
# One model) answers one yes/no question: "does this
# assistant turn forget or contradict any extracted user
# knowledge?" with P(yes) >= 0.5 counted as a knowledge
# attrition. The equation and the LLM-written reason are
# unchanged.
#
# If a Jev call fails, the metric silently defers that
# decision to the LLM.
#
# Hybrid is the cheap/fast lane: the LLM does ONE
# extraction call (not per-turn), and Jev handles the
# per-turn attrition verdicts.

hybrid_knowledge_retention = KnowledgeRetentionMetric(
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still extracts user knowledge
    include_reason=True,
    eval_mode="hybrid",         # Jev judges per-assistant-turn attritions
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same booking conversation as the basic run — every
# fact is recalled.

convo_test_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I'd like to book a table under the name Rivera, party of four, for next Friday at 7:30 PM."),
        Turn(role="assistant", content="Got it — Rivera, party of four, next Friday at 7:30 PM. May I have a phone number for the reservation?"),
        Turn(role="user",      content="Sure, 415-555-0142. And it's my daughter's birthday, so a corner booth would be great if you have one."),
        Turn(role="assistant", content="Perfect, I have a corner booth held for Rivera on Friday at 7:30 PM. We'll note the birthday and have the phone number on file. Anything else?"),
        Turn(role="user",      content="Actually, can we move it to 8:00?"),
        Turn(role="assistant", content="Done — Rivera, party of four, Friday at 8:00 PM in the corner booth. See you then!"),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reports how confident the system_one
# model was on average across its calls — useful for
# debugging borderline cases.

hybrid_knowledge_retention.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_knowledge_retention.score}")
print(f"Confidence: {hybrid_knowledge_retention.confidence}")
print(f"Reason:     {hybrid_knowledge_retention.reason}")
print(f"Passed:     {hybrid_knowledge_retention.is_successful()}")