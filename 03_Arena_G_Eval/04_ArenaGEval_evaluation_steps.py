from basemodel import CustomOpenAI, CustomAWSmodel
from deepeval.test_case import ArenaTestCase, LLMTestCase, SingleTurnParams, Contestant
from deepeval.metrics import ArenaGEval
from deepeval import compare
from rich import print

# ==========================================
# 1. BUILD AN ArenaTestCase WITH TWO CONTESTANTS
# ==========================================
a_test_case = ArenaTestCase(
    contestants=[
        Contestant(
            name="MiniMax",
            hyperparameters={"model": "minimax"},
            test_case=LLMTestCase(
                input="Describe Sasuke Uchiha in one paragraph.",
                actual_output="Sasuke is a ninja from Konoha.",
            ),
        ),
        Contestant(
            name="AWS",
            hyperparameters={"model": "aws"},
            test_case=LLMTestCase(
                input="Describe Sasuke Uchiha in one paragraph.",
                actual_output=(
                    "Sasuke Uchiha is a prodigy of the Uchiha clan, a survivor "
                    "of its massacre, and a member of Team 7 alongside Naruto "
                    "and Sakura. Driven by revenge, he defects from Konoha to "
                    "train under Orochimaru, awakens the Mangekyo Sharingan "
                    "after learning Itachi's truth, and ultimately seeks "
                    "redemption by protecting the Hidden Leaf."
                ),
            ),
        ),
    ]
)

# ==========================================
# 2. DEFINE THE ArenaGEval METRIC USING evaluation_steps
# ==========================================
# You can supply EITHER criteria OR evaluation_steps — never both. Using
# explicit steps gives the judge a deterministic procedure instead of
# letting it infer the rubric from a single sentence.

metric = ArenaGEval(
    name="Character Depth",
    model=CustomOpenAI(),
    evaluation_steps=[
        "Read the input and identify the character being asked about.",
        "Check whether each response correctly identifies the character's name, village, and clan.",
        "Reward responses that mention the character's defining motivations (e.g., revenge, redemption).",
        "Penalize responses that are factually wrong or omit key background.",
        "Choose the contestant that gives the more accurate, complete, and well-written description.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
)

# ==========================================
# 3. COMPARE
# ==========================================
compare(test_cases=[a_test_case], metric=metric)