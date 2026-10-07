from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import RoleAdherenceMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the intermediate steps:
#   - for each assistant turn, the LLM's verdict on
#     whether it stays in character as `chatbot_role`
#   - the reason it gave for that verdict
# This is the fastest way to debug drift — when a bot
# breaks character on turn N, verbose mode surfaces which
# reply and which phrase flagged the break.

role_adherence = RoleAdherenceMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE — middle turn breaks character
# ==========================================
# The first and third assistant turns are in character
# (period slang, clipped tone). The middle one slips into
# modern corporate-speak. Verbose mode will surface the
# specific reply the judge flagged.

convo_test_case = ConversationalTestCase(
    chatbot_role=(
        "A 1920s hardboiled detective who speaks in clipped sentences, "
        "dry wit, and uses period slang like 'dame', 'gumshoe', 'pal', "
        "and 'the joint'. Never breaks character, never uses modern "
        "references or contractions."
    ),
    turns=[
        Turn(role="user",      content="Can you help me find a missing necklace?"),
        Turn(role="assistant", content="The dame came to my office Tuesday, said the sparkler vanished."),       # in character
        Turn(role="user",      content="Where should I start looking?"),
        Turn(role="assistant", content="Sure! I'd recommend leveraging your network and running some analytics on the situation."),  # modern speak — breaks character
        Turn(role="user",      content="Any other ideas?"),
        Turn(role="assistant", content="Tail the husband. The joint he haunts on Sixth Street might talk."),         # in character
    ]
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone prints metric.score and metric.reason directly,
# skipping the report aggregation that `evaluate()` does.
# Verbose output prints above the score.

role_adherence.measure(convo_test_case)
print("---")
print(f"Score:   {role_adherence.score}")          # 2/3 ≈ 0.667
print(f"Reason:  {role_adherence.reason}")
print(f"Passed:  {role_adherence.is_successful()}")  # True (under 0.5 threshold)