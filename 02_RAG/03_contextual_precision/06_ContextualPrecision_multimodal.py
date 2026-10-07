from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase, MLLMImage
from deepeval.metrics import ContextualPrecisionMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC (multimodal)
# ==========================================
# ContextualPrecisionMetric also works on multimodal
# test cases. The LLM judge sees the image(s) inline
# along with the text and applies the same per-node
# relevance + WCP rule against (input, expected_output).

contextual_precision = ContextualPrecisionMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# `input`, `expected_output`, and the relevant
# `retrieval_context` node all carry an MLLMImage
# object. MLLMImage accepts a remote URL, a local path
# (pass `local=True` for file paths), or a base64 data
# URI. The judge treats the image as part of the
# question / evidence when judging each node.

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/375px-Tour_Eiffel_Wikimedia_Commons.jpg"

test_case = LLMTestCase(
    input=f"Tell me about this landmark in France: {MLLMImage(url=image_url)}",
    # Replace with the actual output of your multimodal RAG generator
    actual_output=(
        "This appears to be the Eiffel Tower, which is a famous landmark "
        "in Paris, France."
    ),
    expected_output=f"The Eiffel Tower is located in Paris, France. {MLLMImage(url=image_url)}",
    retrieval_context=[
        f"The Eiffel Tower {MLLMImage(url=image_url)} is a wrought-iron lattice tower built in the late 19th century.",  # relevant
        "Our company was founded in 1998 and is privately owned.",                                                       # unrelated
    ],
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# Standalone form also works:
#   contextual_precision.measure(test_case)
#   print(contextual_precision.score, contextual_precision.reason)

evaluate(test_cases=[test_case], metrics=[contextual_precision])