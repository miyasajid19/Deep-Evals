from basemodel import CustomOpenAI
from deepeval.test_case import SingleTurnParams, LLMTestCase
from deepeval.metrics import GEval
from deepeval import evaluate
from rich import print

clarity = GEval(
    model=CustomOpenAI(),
    name="Clarity",
    evaluation_steps=[
        "Evaluate whether the response uses clear and direct language.",
        "Check if the explanation avoids jargon or explains it when used.",
        "Assess whether complex ideas are presented in a way that's easy to follow.",
        "Identify any vague or confusing parts that reduce understanding."
    ],
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
)

test_cases = [
    LLMTestCase(
        input="Who is Naruto Uzumaki?",
        actual_output="Naruto Uzumaki is the main character of the Naruto series. He is a ninja from the Hidden Leaf Village who dreams of becoming Hokage, the strongest leader of his village."
    ),
    LLMTestCase(
        input="What is the Sharingan?",
        actual_output="The Sharingan is a special eye ability, called a Kekkei Genkai, that some members of the Uchiha clan possess. It allows the user to copy jutsu, predict movements, and cast powerful illusions called Genjutsu."
    ),
    LLMTestCase(
        input="What is the difference between Taijutsu and Ninjutsu?",
        actual_output="Taijutsu refers to physical combat techniques that rely on the body, like punches and kicks, without using chakra. Ninjutsu, on the other hand, uses chakra — the energy inside a ninja's body — to perform special techniques like creating fireballs or summoning creatures."
    ),
]

evaluate(test_cases=test_cases, metrics=[clarity])