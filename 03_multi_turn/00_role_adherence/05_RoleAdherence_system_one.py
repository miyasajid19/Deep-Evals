from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import RoleAdherenceMetric

# ==========================================
# 1. THE METRIC (system_one / Jev-as-a-judge)
# ==========================================
# No LLM is called at all. Jev receives the whole
# conversation and the `chatbot_role` and answers three
# questions:
#
#   Q1 (Noul, weight 2): every assistant turn stays in
#                        character
#   Q2 (Noul, weight 1): no assistant turn breaks
#                        character, even under user pressure
#   Q3 (Score, weight 1): across all turns, how
#                          consistently the role is held
#                          (mostly out → always in)
#
# Final score = weighted mean of the three answers.
#
# Under `system_one` Jev judges the conversation as a
# WHOLE — there's no per-turn scoring like the LLM
# chain does. Pick this mode when you want the cheapest
# possible judge and don't need to compare against an
# LLM-mode score from the same metric.

system_one_role_adherence = RoleAdherenceMetric(
    threshold=0.5,
    include_reason=True,
    eval_mode="system_one",
    # No `model=` needed — Jev runs the metric alone.
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# Same clean role-play as the basic run.

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
# metric.confidence reflects how decisive Jev was. The
# reason lists each question's answer with its
# probability — useful when you want to know *why* Jev
# scored it that way, not just the final number.

system_one_role_adherence.measure(convo_test_case)
print("---")
print(f"Score:      {system_one_role_adherence.score}")
print(f"Confidence: {system_one_role_adherence.confidence}")
print(f"Reason:     {system_one_role_adherence.reason}")
print(f"Passed:     {system_one_role_adherence.is_successful()}")