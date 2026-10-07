from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import RoleAdherenceMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if EVERY assistant turn stays in character
#   - 0 if any single assistant turn breaks character
# It also overrides `threshold` to 1, so is_successful()
# matches the binary score exactly. Use it when any
# out-of-character reply is unacceptable (customer-facing
# brand bots, immersive role-play characters).

strict_role_adherence = RoleAdherenceMetric(
    threshold=0.5,                      # ignored in strict mode
    model=CustomOpenAI(),
    include_reason=True,
    strict_mode=True,
)

# ==========================================
# 2. EVERY TURN IN CHARACTER — expect score 1
# ==========================================
# All three assistant replies use period slang and clipped
# sentences. Strict mode should mark the conversation as
# fully in-character.

perfect_case = ConversationalTestCase(
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
        Turn(role="user",      content="Any other ideas?"),
        Turn(role="assistant", content="Tail the husband. Run his books. The joint he haunts on Sixth Street might talk."),
    ]
)

strict_role_adherence.measure(perfect_case)
print("--- every turn in character ---")
print(f"Score:   {strict_role_adherence.score}")          # 1.0
print(f"Reason:  {strict_role_adherence.reason}")
print(f"Passed:  {strict_role_adherence.is_successful()}")  # True

# ==========================================
# 3. ONE BREAK — expect score 0
# ==========================================
# The middle reply slips into modern corporate-speak.
# Strict mode drops the score to 0. Compare against the
# non-strict `02_verbose` run, where the same conversation
# scores 2/3 ≈ 0.667.

break_case = ConversationalTestCase(
    chatbot_role=(
        "A 1920s hardboiled detective who speaks in clipped sentences, "
        "dry wit, and uses period slang like 'dame', 'gumshoe', 'pal', "
        "and 'the joint'. Never breaks character."
    ),
    turns=[
        Turn(role="user",      content="Can you help me find a missing necklace?"),
        Turn(role="assistant", content="The dame came to my office Tuesday, said the sparkler vanished."),
        Turn(role="user",      content="Where should I start looking?"),
        Turn(role="assistant", content="Sure! I'd recommend leveraging your network and running some analytics on the situation."),  # breaks character
        Turn(role="user",      content="Any other ideas?"),
        Turn(role="assistant", content="Tail the husband. The joint he haunts on Sixth Street might talk."),
    ]
)

strict_role_adherence.measure(break_case)
print("--- one break ---")
print(f"Score:   {strict_role_adherence.score}")          # 0.0
print(f"Reason:  {strict_role_adherence.reason}")
print(f"Passed:  {strict_role_adherence.is_successful()}")  # False