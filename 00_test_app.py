import os
from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval import evaluate
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

class MiniMaxLLM(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

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
        return "MiniMax-M3"

print(os.getenv("MINIMAX_BASE_URL"))
# Instantiate your ChatOpenAI-based MiniMax client
custom_model = ChatOpenAI(
    base_url=os.getenv("MINIMAX_BASE_URL"),
    api_key=os.getenv("MINIMAX_API_KEY"),
    model="Minimax-M3",
    temperature=0.2,
    extra_body={"thinking": {"type": "disabled"}},
)

minimax_llm = MiniMaxLLM(model=custom_model)
import os
from langchain_openai import ChatOpenAI
from deepeval.models import DeepEvalBaseLLM
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval import evaluate
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

class MiniMaxLLM(DeepEvalBaseLLM):
    def __init__(self, model):
        self.model = model

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
        return "MiniMax-M3"

# Instantiate your ChatOpenAI-based MiniMax client
custom_model = ChatOpenAI(
    base_url=os.getenv("MINIMAX_BASE_URL"),
    api_key=os.getenv("MINIMAX_API_KEY"),
    model="Minimax-M3",
    temperature=0.2,
    extra_body={"thinking": {"type": "disabled"}},
)

minimax_llm = MiniMaxLLM(model=custom_model)
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    threshold=0.5,
    model=minimax_llm
)
test_case = LLMTestCase(
    input="I have a persistent cough and fever. Should I be worried?",
    # Replace this with the actual output from your LLM application
    actual_output="A persistent cough and fever could signal various illnesses, from minor infections to more serious conditions like pneumonia or COVID-19. It's advisable to seek medical attention if symptoms worsen, persist beyond a few days, or if you experience difficulty breathing, chest pain, or other concerning signs.",
    expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
)
evaluate([test_case], [correctness_metric])