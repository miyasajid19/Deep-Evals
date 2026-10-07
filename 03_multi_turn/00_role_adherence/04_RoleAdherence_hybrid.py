from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import RoleAdherenceMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid mode for Role Adherence works differently from
# the other hybrid modes: the LLM STILL judges all turns
# in one prompt, but for any turn the LLM considers out
# of character, Jev (a System One model) is asked a
# yes/no "does this turn stay in `chatbot_role`?" with
# P(yes) < 0.5 counting as out of character.
#
# If any Jev call fails, the LLM judges the whole
# conversation instead.
#
# Hybrid is the cheap/fast lane when you suspect the LLM
# is being overly strict — Jev often agrees on the easy
# in-character turns and only diverges on the borderline
# ones.

hybrid_role_adherence = RoleAdherenceMetric(
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still judges in-character turns
    include_reason=True,
    eval_mode="hybrid",         # Jev backstops the borderline ones
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same clean role-play as the basic run. Every assistant
# turn stays in character.

convo_test_case = ConversationalTestCase(
    chatbot_role=(
        "A 1920s hardboiled detective who speaks in clipped sentences, "
        "dry wit, and uses period slang like 'dame', 'gumshoe', 'pal', "
        "and 'the joint'. Never breaks character."
    ),
    turns=[
        Turn(role="user",      content="Can you help me find a missing necklace?"),
        Turn(role="assistant", content="The dame came to my office Tuesday, said the sparkler vanished."),
        Turn(role="user",      content="Where should I start looking?"),
        Turn(role="assistant", content="Tip the butler, pal. The butler always knows where the ice is hidden."),
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# metric.confidence reports how confident the system_one
# model was on average across its calls — useful for
# debugging borderline cases.

hybrid_role_adherence.measure(convo_test_case)
print("---")
print(f"Score:      {hybrid_role_adherence.score}")
print(f"Confidence: {hybrid_role_adherence.confidence}")
print(f"Reason:     {hybrid_role_adherence.reason}")
print(f"Passed:     {hybrid_role_adherence.is_successful()}")