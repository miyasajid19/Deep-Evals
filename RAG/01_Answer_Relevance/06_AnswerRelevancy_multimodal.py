from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase, MLLMImage
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC (multimodal)
# ==========================================
# AnswerRelevancyMetric also works on multimodal test
# cases. The LLM judge sees the image(s) inline as well
# the text and applies the same statement-relevancy rule:
# are the statements made in `actual_output` addressing
# the multimodal `input`?

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# `input` is a string that contains an MLLMImage object.
# MLLMImage accepts a remote URL, a local path, or a
# base64 data URI. The judge will treat the image as
# part of the question when deciding which statements
# in `actual_output` are relevant.

test_cases = [
    LLMTestCase(
        input=f"who is he? : {MLLMImage(url='https://media.cnn.com/api/v1/images/stellar/prod/gettyimages-2234200789.jpg')}",
        actual_output="He is Ronaldo, the greatest player of all time. G.O.A.T.",
    ),
    LLMTestCase(
        input=f"{MLLMImage(url=r'C:/Users/miyas/Downloads/savira.png', local=True)}",
        actual_output="It is a company logo",
    ),
]

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# Standalone form also works:
#   answer_relevancy.measure(test_case)
#   print(answer_relevancy.score, answer_relevancy.reason)

evaluate(test_cases=test_cases, metrics=[answer_relevancy])