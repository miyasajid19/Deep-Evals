from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

test_case = LLMTestCase(input="what are capital city of nepal?", actual_output="The capital city of nepal is ktm.", expected_output="Kathmandu is the capital city of Nepal.However the old capital city of Nepal was Gorkha.")
correctness = GEval(
    model=CustomOpenAI(),
    name="Correctness",
    criteria="Correctness - determine if the actual output is correct according to the expected output.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    strict_mode=True
)

correctness.measure(test_case)
print(correctness.score, correctness.reason)