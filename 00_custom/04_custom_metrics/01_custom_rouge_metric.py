from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval.scorer import Scorer
from rich import print

# ==========================================
# 1. DEFINE A NON-LLM CUSTOM METRIC
# ==========================================
# Custom metrics don't have to use an LLM. This one scores a response
# against an expected output using ROUGE-1 F1 — no model inference needed.
# Drop `include_reason` and `evaluation_model`: there's no LLM to call.

class RougeMetric(BaseMetric):
    def __init__(self, threshold: float = 0.5, score_type: str = "rouge1"):
        self.threshold = threshold
        self.score_type = score_type
        self.scorer = Scorer()

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            self.score = self.scorer.rouge_score(
                prediction=test_case.actual_output,
                target=test_case.expected_output,
                score_type=self.score_type,
            )
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        # No async ROUGE implementation — reuse the sync method.
        return self.measure(test_case)

    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            try:
                self.success = self.score >= self.threshold
            except TypeError:
                self.success = False
        return self.success

    @property
    def __name__(self):
        return "Rouge Metric"

# ==========================================
# 2. EXAMPLE USAGE
# ==========================================
metric = RougeMetric(threshold=0.5, score_type="rouge1")

test_cases = [
    LLMTestCase(
        input="Who is the Fourth Hokage?",
        actual_output="The Fourth Hokage is Minato Namikaze.",
        expected_output="Minato Namikaze is the Fourth Hokage, also known as the Yellow Flash.",
    ),
    LLMTestCase(
        input="What is sealed inside Naruto?",
        actual_output="Kurama is sealed inside him.",
        expected_output="Kurama, the Nine-Tails beast, is sealed inside Naruto Uzumaki.",
    ),
]

for tc in test_cases:
    metric.measure(tc)
    print(f"Score:    {metric.score:.3f}")
    print(f"Passed:   {metric.is_successful()}")
    print("-" * 40)