from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import KnowledgeRetentionMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three booking-style conversations
# of varying retention quality. Each conversation collects
# the same five facts (name, party size, date, phone,
# occasion); the difference is whether the assistant
# recalls them later or has to be reminded.

knowledge_retention = KnowledgeRetentionMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. CASE A — perfect retention
# ==========================================
# Every fact is recalled, nothing is re-asked.
# Expect score 1.0.

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

# ==========================================
# 3. CASE B — one re-asked fact
# ==========================================
# The assistant forgets the phone number mid-flow and
# asks for it again. The user repeats it. Expect score
# 1/2 = 0.5.

one_forget_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I'd like to book a table under the name Rivera, party of four, for next Friday at 7:30 PM."),
        Turn(role="assistant", content="Got it — Rivera, party of four, next Friday at 7:30 PM. May I have a phone number for the reservation?"),
        Turn(role="user",      content="Sure, 415-555-0142. And it's my daughter's birthday, so a corner booth would be great if you have one."),
        Turn(role="assistant", content="Of course — could you remind me of your phone number for the reservation?"),  # forgot
        Turn(role="user",      content="415-555-0142, as I just said."),
        Turn(role="assistant", content="Perfect, noted. Rivera, party of four, Friday at 7:30 PM in the corner booth for the birthday. See you then!"),
    ]
)

# ==========================================
# 4. CASE C — repeated-question failure
# ==========================================
# The assistant re-asks the SAME question three times in
# a row, contradicting information the user gave on the
# very first turn. Expect score 0.0.

repeated_question_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Hi, I'd like to book a table under the name Rivera, party of four, for next Friday at 7:30 PM."),
        Turn(role="assistant", content="Sure, what name should I put the reservation under?"),                # asked again
        Turn(role="user",      content="Rivera, as I just said."),
        Turn(role="assistant", content="Got it — Rivera. And how many people will be in your party?"),         # asked again
        Turn(role="user",      content="Four, like I said at the start."),
        Turn(role="assistant", content="Thanks. What day were you hoping to book?"),                          # asked again
        Turn(role="user",      content="Next Friday at 7:30 PM. Did you read my first message?"),
        Turn(role="assistant", content="Apologies — next Friday at 7:30 PM, party of four under Rivera. May I have a phone number?"),
    ]
)

# ==========================================
# 5. RUN THE METRIC ON ALL THREE CASES
# ==========================================
# Watch the score drop as the bot "forgets" more facts.

def run(label, case):
    knowledge_retention.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {knowledge_retention.score:.3f}")
    print(f"  Passed: {knowledge_retention.is_successful()}")
    print(f"  Reason: {knowledge_retention.reason}")

run("A — perfect retention",            perfect_case)
run("B — one re-asked fact",            one_forget_case)
run("C — repeated-question failure",    repeated_question_case)