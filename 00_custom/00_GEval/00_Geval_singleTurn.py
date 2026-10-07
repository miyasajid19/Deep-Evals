from basemodel import CustomOpenAI
from deepeval.test_case import SingleTurnParams,LLMTestCase
from deepeval.metrics import GEval
from deepeval import evaluate
from rich import print

correctness=GEval(
    model=CustomOpenAI(),
    name="Correctness",
    criteria="Correctness - determine if the actual output is correct according to the expected output.",
     evaluation_steps=[
        "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
        "You should also heavily penalize omission of detail",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
)


faithfulness=GEval(
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

test_case=[
    LLMTestCase(input="what is the  capital city of nepal?", actual_output="The capital city of nepal is kathmandu.", expected_output="Kathmandu is the capital city of Nepal."),
    LLMTestCase(input="what is the old capital city of nepal?",actual_output="The Old capital city of nepal was Gorkha.",expected_output="The old capital city of Nepal was Gorkha."),
]


evaluate(test_cases=test_case, metrics=[correctness, faithfulness])