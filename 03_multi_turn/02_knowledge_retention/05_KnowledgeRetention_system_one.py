from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import KnowledgeRetentionMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation and answers three questions:
#
#   Q1 (Noul, weight 2): no assistant turn forgets or
#                        contradicts user-stated facts
#   Q2 (Noul, weight 1): no assistant turn re-asks for
#                        info the user already gave
#                        (clarification doesn't count)
#   Q3 (Score, weight 1): across the conversation, how
#                          well facts are retained
#                          (forgets most → retains all)
#
# Final score = weighted mean of the three answers.
#
# system_one mode is the right pick when you want the
# cheapest possible judge and don't need to compare
# against an LLM-mode score from the same metric.

system_one_knowledge_retention = KnowledgeRetentionMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same booking conversation as the basic run.

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
# metric.confidence reflects how decisive Jev was. The
# reason lists each question's answer with its
# probability — useful when you want to know *why* Jev
# scored it that way, not just the final number.

system_one_knowledge_retention.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_knowledge_retention.score}")
print(f"Confidence: {system_one_knowledge_retention.confidence}")
print(f"Reason:     {system_one_knowledge_retention.reason}")
print(f"Passed:     {system_one_knowledge_retention.is_successful()}")