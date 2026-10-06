from basemodel import CustomOpenAI
from deepeval.test_case import SingleTurnParams, LLMTestCase
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval import evaluate
from rich import print

correctness = GEval(
    model=CustomOpenAI(),
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    rubric=[
        Rubric(score_range=(0, 2), expected_outcome="Factually incorrect."),
        Rubric(score_range=(3, 6), expected_outcome="Mostly correct."),
        Rubric(score_range=(7, 9), expected_outcome="Correct but missing minor details."),
        Rubric(score_range=(10, 10), expected_outcome="100% correct."),
    ]
)

faithfulness = GEval(
    model=CustomOpenAI(),
    name="Faithfulness",
    criteria="Faithfulness - determine if the actual output is faithful to the input.",
    evaluation_steps=[
        "Check if the 'actual output' contains information that is not present in the 'input'",
        "Ensure that the 'actual output' does not introduce any new information that is not supported by the 'input'",
        "Verify that the 'actual output' is a direct consequence of the 'input'"
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
)

test_cases = [
    LLMTestCase(
        input="Who is the Fourth Hokage in Naruto?",
        actual_output="The Fourth Hokage in Naruto is Minato Namikaze.",
        expected_output="Minato Namikaze is the Fourth Hokage in Naruto, also known as the Yellow Flash."
    ),
    LLMTestCase(
        input="What is the Nine-Tails beast sealed inside Naruto Uzumaki?",
        actual_output="The Nine-Tails beast sealed inside Naruto Uzumaki is Kurama.",
        expected_output="The Nine-Tails beast sealed inside Naruto Uzumaki is Kurama, a powerful tailed beast."
    ),
    LLMTestCase(
        input="What village does Naruto Uzumaki belong to?",
        actual_output="Naruto Uzumaki belongs to the Hidden Leaf Village, also known as Konohagakure.",
        expected_output="Naruto Uzumaki belongs to the Hidden Leaf Village, known as Konohagakure."
    ),
]

evaluate(test_cases=test_cases, metrics=[correctness, faithfulness])