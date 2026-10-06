from basemodel import CustomOpenAI
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from rich import print

# Define correctness metric
correctness_metric = GEval(
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

# Simulated LLM call (your actual LLM logic goes here)
def call_llm(input: str) -> str:
    responses = {
        "Who is the Fourth Hokage?": "The Fourth Hokage is Minato Namikaze, also known as the Yellow Flash.",
        "What is sealed inside Naruto?": "Kurama, the Nine-Tails beast, is sealed inside Naruto.",
        "What village is Naruto from?": "Naruto is from the Hidden Leaf Village, known as Konohagakure.",
    }
    return responses.get(input, "I don't know.")

# Inner component — scored by correctness_metric
@observe(metrics=[correctness_metric])
def inner_component(input: str, expected_output: str):
    actual_output = call_llm(input)
    update_current_span(
        test_case=LLMTestCase(
            input=input,
            actual_output=actual_output,
            expected_output=expected_output,
        )
    )
    return actual_output

# Top-level LLM app
@observe()
def llm_app(input: str, expected_output: str):
    result = inner_component(input, expected_output)
    return result

# Dataset with Naruto-themed goldens
dataset = EvaluationDataset(goldens=[
    Golden(
        input="Who is the Fourth Hokage?",
        expected_output="Minato Namikaze is the Fourth Hokage, also known as the Yellow Flash."
    ),
    Golden(
        input="What is sealed inside Naruto?",
        expected_output="Kurama, the Nine-Tails beast, is sealed inside Naruto Uzumaki."
    ),
    Golden(
        input="What village is Naruto from?",
        expected_output="Naruto is from the Hidden Leaf Village, known as Konohagakure."
    ),
])

# Evaluation loop
for golden in dataset.evals_iterator():
    llm_app(golden.input, golden.expected_output)