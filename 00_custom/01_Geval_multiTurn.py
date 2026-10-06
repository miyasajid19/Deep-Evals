from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MultiTurnParams
from deepeval.metrics import ConversationalGEval
from deepeval import evaluate
from rich import print

correctness = ConversationalGEval(
    model=CustomOpenAI(),
    name="Correctness",
    evaluation_steps=[
        "Check whether the facts stated by the assistant across all turns contradict the 'expected outcome'",
        "Heavily penalize the conversation if the assistant never reaches the 'expected outcome' by the final turn",
        "Vague language, or contradicting OPINIONS, are OK"
    ],
    evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.EXPECTED_OUTCOME],
)

faithfulness = ConversationalGEval(
    model=CustomOpenAI(),
    name="Faithfulness",
    evaluation_steps=[
        "Check if any assistant turn contains information that is not present in the user's messages",
        "Ensure that the assistant does not introduce new information not supported by the conversation",
        "Verify that each assistant response is a direct consequence of the user's input in that turn"
    ],
    evaluation_params=[MultiTurnParams.CONTENT],
)

test_cases = [
    ConversationalTestCase(
        scenario="User asking about the capital city of Nepal.",
        expected_outcome="The assistant correctly identifies Kathmandu as the capital city of Nepal.",
        turns=[
            Turn(role="user", content="What is the capital city of Nepal?"),
            Turn(role="assistant", content="The capital city of Nepal is Kathmandu."),
        ]
    ),
    ConversationalTestCase(
        scenario="User asking about the old capital city of Nepal.",
        expected_outcome="The assistant correctly identifies Gorkha as the old capital city of Nepal before unification. But before unification, there was no country called Nepal.",
        turns=[
            Turn(role="user", content="What is the old capital city of Nepal?"),
            Turn(role="assistant", content="Kathmandu is the only capital city of Nepal since unification."),
            Turn(role="user", content="What is the old capital city of Nepal before unification?"),
            Turn(role="assistant", content="Before unification,there was not country called Nepal. However, modern nepal unification started from gorkha, so the old capital city of Nepal was Gorkha."),
        ]
    ),
]

evaluate(test_cases=test_cases, metrics=[correctness, faithfulness])