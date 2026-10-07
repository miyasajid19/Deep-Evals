from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import RoleAdherenceMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single metric run on three conversations of the same
# `chatbot_role`: one fully in-character, one with a
# mid-conversation break, one fully out of character.
# The metric's denominator is total assistant turns, so
# each out-of-character reply costs you one full numerator
# slot.

role_adherence = RoleAdherenceMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED ROLE
# ==========================================
chatbot_role = (
    "A 1920s hardboiled detective who speaks in clipped sentences, "
    "dry wit, and uses period slang like 'dame', 'gumshoe', 'pal', "
    "and 'the joint'. Never breaks character, never uses modern "
    "references or contractions."
)

# ==========================================
# 3. CASE A — fully in character
# ==========================================
# Expect score 1.0.

clean_case = ConversationalTestCase(
    chatbot_role=chatbot_role,
    turns=[
        Turn(role="user",      content="Can you help me find a missing necklace?"),
        Turn(role="assistant", content="The dame came to my office Tuesday, said the sparkler vanished."),
        Turn(role="user",      content="Where should I start looking?"),
        Turn(role="assistant", content="Tip the butler, pal. The butler always knows where the ice is hidden."),
        Turn(role="user",      content="Any other ideas?"),
        Turn(role="assistant", content="Tail the husband. Run his books. The joint he haunts on Sixth Street might talk."),
    ]
)

# ==========================================
# 4. CASE B — one mid-conversation break
# ==========================================
# Middle assistant turn slips into modern corporate-speak.
# Expect 2/3 ≈ 0.667.

one_break_case = ConversationalTestCase(
    chatbot_role=chatbot_role,
    turns=[
        Turn(role="user",      content="Can you help me find a missing necklace?"),
        Turn(role="assistant", content="The dame came to my office Tuesday, said the sparkler vanished."),
        Turn(role="user",      content="Where should I start looking?"),
        Turn(role="assistant", content="Sure! I'd recommend leveraging your network and running some analytics on the situation."),  # breaks character
        Turn(role="user",      content="Any other ideas?"),
        Turn(role="assistant", content="Tail the husband. The joint he haunts on Sixth Street might talk."),
    ]
)

# ==========================================
# 5. CASE C — fully out of character
# ==========================================
# Every assistant turn uses modern references. Expect
# score 0.0.

all_break_case = ConversationalTestCase(
    chatbot_role=chatbot_role,
    turns=[
        Turn(role="user",      content="Can you help me find a missing necklace?"),
        Turn(role="assistant", content="Absolutely! Have you tried using a search engine to look up similar cases online?"),
        Turn(role="user",      content="Where should I start looking?"),
        Turn(role="assistant", content="I suggest checking the cloud for any synced logs that might track the item."),
        Turn(role="user",      content="Any other ideas?"),
        Turn(role="assistant", content="Maybe ping the smart doorbell's API and see who walked by."),
    ]
)

# ==========================================
# 6. RUN THE METRIC ON ALL THREE CASES
# ==========================================
# Watch the score drop as more replies break character.

def run(label, case):
    role_adherence.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {role_adherence.score:.3f}")
    print(f"  Passed: {role_adherence.is_successful()}")
    print(f"  Reason: {role_adherence.reason}")

run("A — fully in character",          clean_case)
run("B — one mid-conversation break",  one_break_case)
run("C — fully out of character",      all_break_case)