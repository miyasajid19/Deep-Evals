import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

load_dotenv()

# Step 1: Define your custom LLM class
class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOpenAI(
            model_name=os.getenv("MINIMAX_MODEL"),
            base_url=os.getenv("MINIMAX_BASE_URL"),
            api_key=os.getenv("MINIMAX_API_KEY"),
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom OpenAI-Compatible Model"


# Step 2: Instantiate your custom LLM
custom_llm = CustomOpenAI()


# Step 3: Define and run the test
def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also lightly penalize omission of detail, and focus on the main idea",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        threshold=0.5,
        model=custom_llm
    )

    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don't improve in a few days.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
    )

    # Measure the metric manually to print results
    correctness_metric.measure(test_case)

    # Print the results
    print("===========================================")
    print(f"Metric Name  : {correctness_metric.name}")
    print(f"Score        : {correctness_metric.score}")
    print(f"Threshold    : {correctness_metric.threshold}")
    print(f"Passed       : {correctness_metric.score >= correctness_metric.threshold}")
    print(f"Reason       : {correctness_metric.reason}")
    print("===========================================")

    # Also run assert_test to enforce pass/fail
    assert_test(test_case, [correctness_metric])


if __name__ == "__main__":
    test_correctness()