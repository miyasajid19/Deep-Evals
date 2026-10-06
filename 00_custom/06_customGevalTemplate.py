from basemodel import CustomOpenAI
from deepeval.test_case import SingleTurnParams, LLMTestCase
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import GEvalTemplate
from deepeval import evaluate
from rich import print
import textwrap

# ==========================================
# 1. DEFINE CUSTOM TEMPLATE
# ==========================================
class CustomGEvalTemplate(GEvalTemplate):
    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str):
        return textwrap.dedent(
            f"""
            You are given evaluation criteria for assessing {parameters}. Based on the criteria,
            produce 3-4 clear steps that explain how to evaluate the quality of {parameters}.

            Criteria:
            {criteria}

            Return JSON only, in this format:
            {{
                "steps": [
                    "Step 1",
                    "Step 2",
                    "Step 3"
                ]
            }}

            JSON:
            """
        )

# ==========================================
# 2. DEFINE METRICS WITH CUSTOM TEMPLATE
# ==========================================
correctness = GEval(
    model=CustomOpenAI(),
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    evaluation_template=CustomGEvalTemplate,
)

clarity = GEval(
    model=CustomOpenAI(),
    name="Clarity",
    criteria="Determine whether the actual output is written in a clear and easy to understand manner.",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    evaluation_template=CustomGEvalTemplate,
)

# ==========================================
# 3. NARUTO-THEMED TEST CASES
# ==========================================
test_cases = [
    LLMTestCase(
        input="Who is the Fourth Hokage?",
        actual_output="The Fourth Hokage is Minato Namikaze, also known as the Yellow Flash of the Leaf.",
        expected_output="Minato Namikaze is the Fourth Hokage, famous for his speed and known as the Yellow Flash."
    ),
    LLMTestCase(
        input="What is the Sharingan?",
        actual_output="The Sharingan is a special eye technique used by the Uchiha clan that lets them copy jutsu and see through illusions.",
        expected_output="The Sharingan is a Kekkei Genkai of the Uchiha clan that grants the ability to copy techniques and cast Genjutsu."
    ),
    LLMTestCase(
        input="What is Kurama?",
        actual_output="Kurama is the Nine-Tails fox beast sealed inside Naruto Uzumaki.",
        expected_output="Kurama is the Nine-Tails, a powerful tailed beast that was sealed inside Naruto Uzumaki by his father Minato."
    ),
]

# ==========================================
# 4. EVALUATE
# ==========================================
evaluate(test_cases=test_cases, metrics=[correctness, clarity])