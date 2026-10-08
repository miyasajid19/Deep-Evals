from basemodel import CustomOpenAI
from deepeval.classifiers import Classifier, Label
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

classifier = Classifier(
    name="topic",
    model=CustomOpenAI(),
    include_reason=True,
    labels=[
        Label(name="billing", description="Questions about invoices, charges, or payment methods."),
        Label(name="refund", description="Requests to get money back for a purchase that has already been made."),
        Label(name="shipping", description="Questions about delivery status, times, or addresses."),
    ],
)

sentiment = Classifier(
    name="sentiment",
    model=CustomOpenAI(),
    include_reason=True,
    labels=[
        Label(name="positive", description="The user is happy, satisfied, or pleased with the service."),
        Label(name="negative", description="The user is unhappy, dissatisfied, or displeased with the service."),
        Label(name="neutral", description="The user is neither happy nor unhappy; they are indifferent or have no strong feelings."),
    ],
)

test_cases = [
    LLMTestCase(
        input="I was charged twice this month.",
        actual_output="Sorry about that. I can see the duplicate charge and will reverse it.",
        expected_labels={classifier.name: "billing", sentiment.name: "negative"},
    ),
    LLMTestCase(
        input="Where is my order?",
        actual_output="Your order is on its way and will arrive by Friday.",
        expected_labels={classifier.name: "shipping", sentiment.name: "neutral"},
    ),
    LLMTestCase(
        input="I want a refund for my last purchase.",
        actual_output="I've initiated a refund for your last order. It will appear in 3-5 business days.",
        expected_labels={classifier.name: "refund", sentiment.name: "neutral"},
    ),
    LLMTestCase(
        input="Thank you, everything was perfect!",
        actual_output="We're glad to hear that! Thank you for your kind words.",
        expected_labels={classifier.name: "billing", sentiment.name: "positive"},
    ),
]

evaluate(test_cases=test_cases, classifiers=[classifier, sentiment])