from basemodel import CustomOpenAI
from deepeval.test_case import ArenaTestCase, LLMTestCase, SingleTurnParams, Contestant
from deepeval.metrics import ArenaGEval
from rich import print

# ==========================================
# 1. BUILD THE ArenaTestCase
# ==========================================
a_test_case = ArenaTestCase(
    contestants=[
        Contestant(
            name="Concise",
            hyperparameters={"style": "concise"},
            test_case=LLMTestCase(
                input="Explain the Sharingan.",
                actual_output="The Sharingan is an eye power of the Uchiha clan.",
            ),
        ),
        Contestant(
            name="Detailed",
            hyperparameters={"style": "detailed"},
            test_case=LLMTestCase(
                input="Explain the Sharingan.",
                actual_output=(
                    "The Sharingan is a Kekkei Genkai — a bloodline limit — "
                    "that awakens in members of the Uchiha clan under strong "
                    "emotional stress. It grants the user heightened "
                    "perception, the ability to copy jutsu, predict movement, "
                    "and cast powerful illusions called Genjutsu."
                ),
            ),
        ),
    ]
)

# ==========================================
# 2. DEFINE THE METRIC
# ==========================================
metric = ArenaGEval(
    name="Clarity Winner",
    criteria="Choose the contestant whose response is clearer and easier to follow.",
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
# Standalone execution is useful for debugging, but it does NOT include
# the testing reports, Confident AI integration, or speed/cost
# optimizations that `compare()` offers.

metric.measure(a_test_case)
print(f"Winner:  {metric.winner}")
print(f"Reason:  {metric.reason}")