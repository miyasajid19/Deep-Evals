from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase, MLLMImage
from deepeval.metrics import FaithfulnessMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC (multimodal)
# ==========================================
# FaithfulnessMetric also works on multimodal test
# cases. The LLM judge sees the image(s) inline along
# with the text and applies the same claim-vs-context
# rule: are the claims in `actual_output` grounded in
# `retrieval_context` (text or image)?

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
# `input` and `retrieval_context` both carry an
# MLLMImage object. MLLMImage accepts a remote URL, a
# local path (pass `local=True` for file paths), or a
# base64 data URI. The judge treats the image as part
# of the question / evidence when judging faithfulness.

test_case = LLMTestCase(
    input=f"Tell me about this landmark in France: {MLLMImage(url='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/375px-Tour_Eiffel_Wikimedia_Commons.jpg')}",
    # Replace with the actual output of your multimodal RAG generator
    actual_output=(
        "This appears to be the Eiffel Tower, which is a famous landmark "
        "in Paris, France. It was completed in 1889."
    ),
    retrieval_context=[
        f"The Eiffel Tower {MLLMImage(url='https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/375px-Tour_Eiffel_Wikimedia_Commons.jpg')} is a wrought-iron lattice tower built in the late 19th century.",
        "The Eiffel Tower is located in Paris, France.",
        "Construction of the Eiffel Tower was completed in 1889.",
    ],
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# Standalone form also works:
#   faithfulness.measure(test_case)
#   print(faithfulness.score, faithfulness.reason)

evaluate(test_cases=[test_case], metrics=[faithfulness])