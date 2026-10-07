from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MultiTurnParams
from deepeval.metrics import ConversationalGEval
from deepeval.metrics.conversational_g_eval import ConversationalGEvalTemplate
from deepeval import evaluate
from rich import print
import textwrap

# ==========================================
# 1. DEFINE CUSTOM TEMPLATE
# ==========================================
class CustomConvoGEvalTemplate(ConversationalGEvalTemplate):
    @staticmethod
    def generate_evaluation_steps(parameters: str, criteria: str):
        return textwrap.dedent(
            f"""
            You are given criteria for evaluating a conversation based on the following parameters: {parameters}.
            Write 3-4 clear and concise evaluation steps that describe how to judge the quality of each turn and the conversation overall.

            Criteria:
            {criteria}

            Return JSON only in the format:
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
correctness = ConversationalGEval(
    model=CustomOpenAI(),
    name="Correctness",
    criteria="Determine whether the assistant provides factually correct information about Naruto across all turns.",
    evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.EXPECTED_OUTCOME],
    evaluation_template=CustomConvoGEvalTemplate,
)

empathy = ConversationalGEval(
    model=CustomOpenAI(),
    name="Empathy",
    criteria="Evaluate whether the assistant shows genuine understanding and empathy when the user expresses confusion or frustration.",
    evaluation_params=[MultiTurnParams.CONTENT],
    evaluation_template=CustomConvoGEvalTemplate,
)

# ==========================================
# 3. NARUTO-THEMED MULTI-TURN TEST CASES
# ==========================================
test_cases = [
    ConversationalTestCase(
        scenario="User asking about the relationship between Naruto and Kurama.",
        expected_outcome="The assistant correctly explains that Kurama is the Nine-Tails beast sealed inside Naruto by his father Minato.",
        turns=[
            Turn(role="user", content="Who is Kurama?"),
            Turn(role="assistant", content="Kurama is the Nine-Tails, one of the most powerful tailed beasts in the Naruto universe."),
            Turn(role="user", content="What is his connection to Naruto?"),
            Turn(role="assistant", content="Kurama was sealed inside Naruto Uzumaki at birth by his father, the Fourth Hokage Minato Namikaze, to protect the village from Kurama's destructive power."),
        ]
    ),
    ConversationalTestCase(
        scenario="User confused about the difference between Taijutsu and Ninjutsu.",
        expected_outcome="The assistant clearly explains that Taijutsu is physical combat without chakra, while Ninjutsu uses chakra to perform special techniques.",
        turns=[
            Turn(role="user", content="I don't understand the difference between Taijutsu and Ninjutsu. Can you help?"),
            Turn(role="assistant", content="Of course! Taijutsu is purely physical combat — think punches, kicks, and body movements — with no chakra involved."),
            Turn(role="user", content="And Ninjutsu?"),
            Turn(role="assistant", content="Ninjutsu uses chakra, the energy inside a ninja's body, to perform special techniques like fireballs, shadow clones, or summoning creatures."),
        ]
    ),
    ConversationalTestCase(
        scenario="User asking about the Uchiha clan and the Sharingan.",
        expected_outcome="The assistant correctly identifies the Sharingan as a Kekkei Genkai of the Uchiha clan that allows copying jutsu and casting Genjutsu.",
        turns=[
            Turn(role="user", content="What is special about the Uchiha clan?"),
            Turn(role="assistant", content="The Uchiha clan is one of the most powerful clans in the Hidden Leaf Village, known for their exceptional combat skills and their unique eye technique."),
            Turn(role="user", content="What eye technique?"),
            Turn(role="assistant", content="It's called the Sharingan — a Kekkei Genkai that grants the ability to copy any jutsu they see, predict enemy movements, and cast powerful illusions called Genjutsu."),
        ]
    ),
]

# ==========================================
# 4. EVALUATE
# ==========================================
evaluate(test_cases=test_cases, metrics=[correctness, empathy])