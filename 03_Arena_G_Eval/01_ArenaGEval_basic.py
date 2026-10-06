from basemodel import CustomOpenAI, CustomAWSmodel
from deepeval.test_case import ArenaTestCase, LLMTestCase, SingleTurnParams, Contestant
from deepeval.metrics import ArenaGEval
from deepeval import compare
from rich import print

# ==========================================
# 1. BUILD THREE ArenaTestCases WITH TWO CONTESTANTS EACH
# ==========================================

test_cases = [
    ArenaTestCase(
        contestants=[
            Contestant(
                name="MiniMax",
                hyperparameters={"model": "minimax"},
                test_case=LLMTestCase(
                    input="Who is the Fourth Hokage in Naruto?",
                    actual_output="The Fourth Hokage is Minato Namikaze.",
                ),
            ),
            Contestant(
                name="AWS",
                hyperparameters={"model": "aws"},
                test_case=LLMTestCase(
                    input="Who is the Fourth Hokage in Naruto?",
                    actual_output=(
                        "The Fourth Hokage of the Hidden Leaf Village is "
                        "Minato Namikaze, famously known as the Yellow Flash of "
                        "the Leaf for his unmatched speed in battle."
                    ),
                ),
            ),
        ]
    ),
    ArenaTestCase(
        contestants=[
            Contestant(
                name="MiniMax",
                hyperparameters={"model": "minimax"},
                test_case=LLMTestCase(
                    input="What is the name of Naruto's son?",
                    actual_output="Naruto's son is Boruto Uzumaki.",
                ),
            ),
            Contestant(
                name="AWS",
                hyperparameters={"model": "aws"},
                test_case=LLMTestCase(
                    input="What is the name of Naruto's son?",
                    actual_output=(
                        "Naruto Uzumaki's son is Boruto Uzumaki, the main "
                        "protagonist of the sequel series 'Boruto: Naruto Next "
                        "Generations'. He inherited his father's determination "
                        "and his mother Hinata's gentle eyes."
                    ),
                ),
            ),
        ]
    ),
    ArenaTestCase(
        contestants=[
            Contestant(
                name="MiniMax",
                hyperparameters={"model": "minimax"},
                test_case=LLMTestCase(
                    input="What is the Nine-Tails' real name?",
                    actual_output="The Nine-Tails' real name is Kurama.",
                ),
            ),
            Contestant(
                name="AWS",
                hyperparameters={"model": "aws"},
                test_case=LLMTestCase(
                    input="What is the Nine-Tails' real name?",
                    actual_output=(
                        "The Nine-Tails Fox sealed inside Naruto is named "
                        "Kurama. Over time, Naruto and Kurama develop a deep "
                        "bond of friendship and trust, making them one of the "
                        "most powerful duos in the series."
                    ),
                ),
            ),
        ]
    ),
]

# ==========================================
# 2. DEFINE THE ArenaGEval METRIC
# ==========================================

metric = ArenaGEval(
    name="Factual Friendliness",
    model=CustomOpenAI(),
    criteria=(
        "Choose the winner based on the response's factual accuracy and "
        "overall friendliness to the user. Penalize incorrect facts; reward "
        "responses that are warm and informative."
    ),
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
    ],
)

# ==========================================
# 3. COMPARE
# ==========================================
compare(test_cases=test_cases, metric=metric)