from basemodel import CustomOpenAI
from deepeval.test_case import ArenaTestCase, LLMTestCase, SingleTurnParams, Contestant
from deepeval.metrics import ArenaGEval
from deepeval import compare
from rich import print

# ==========================================
# 1. BUILD AN ArenaTestCase WITH MULTIPLE CONTESTANTS
# ==========================================
# An ArenaTestCase can hold more than two contestants. ArenaGEval picks a
# single overall winner — the one that beats every other head-to-head
# under blinded, randomized positioning.

a_test_case = ArenaTestCase(
    contestants=[
        Contestant(
            name="MiniMax",
            hyperparameters={"model": "minimax"},
            test_case=LLMTestCase(
                input="Write a short poem about Naruto Uzumaki.",
                actual_output="Naruto runs and never quits.",
            ),
        ),
        Contestant(
            name="AWS",
            hyperparameters={"model": "aws"},
            test_case=LLMTestCase(
                input="Write a short poem about Naruto Uzumaki.",
                actual_output=(
                    "Orange jacket, bright as the dawn,\n"
                    "Nine-tailed fox that once was drawn.\n"
                    "From outcast boy to village flame,\n"
                    "Uzumaki earned his Hokage name."
                ),
            ),
        ),
        # Contestant(
        #     name="Claude",
        #     hyperparameters={"model": "claude-4"},
        #     test_case=LLMTestCase(
        #         input="Write a short poem about Naruto Uzumaki.",
        #         actual_output=(
        #             "A ninja boy with golden dreams,\n"
        #             "believes in things that no one sees."
        #         ),
        #     ),
        # ),
    ]
)

# ==========================================
# 2. DEFINE THE ArenaGEval METRIC
# ==========================================
metric = ArenaGEval(
    model=CustomOpenAI(),
    name="Best Poem",
    criteria=(
        "Choose the contestant whose poem best captures the spirit of "
        "Naruto Uzumaki — his determination, optimism, and growth from "
        "outcast to hero."
    ),
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
)

# ==========================================
# 3. COMPARE
# ==========================================
compare(test_cases=[a_test_case], metric=metric)