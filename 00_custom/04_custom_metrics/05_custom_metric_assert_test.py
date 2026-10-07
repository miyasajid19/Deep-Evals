from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from deepeval import assert_test
from rich import print

# ==========================================
# 1. DEFINE A CUSTOM METRIC
# ==========================================
# A simple length-based metric: pass only when the actual output is
# between `min_length` and `max_length` (inclusive). Deterministic, fast,
# and useful for catching "the model produced an empty / wall-of-text reply".

class LengthMetric(BaseMetric):
    def __init__(self, min_length: int = 1, max_length: int = 500, threshold: float = 0.5):
        self.threshold = threshold
        self.min_length = min_length
        self.max_length = max_length

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            n = len((test_case.actual_output or "").split())
            in_range = self.min_length <= n <= self.max_length
            # Map length to a 0..1 score: 1.0 if in range, otherwise a
            # linear penalty the further outside the bounds we drift.
            if in_range:
                self.score = 1.0
            else:
                distance = min(abs(n - self.min_length), abs(n - self.max_length))
                self.score = max(0.0, 1.0 - distance / 50)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
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
        return "Length Metric"

# ==========================================
# 2. USE THE CUSTOM METRIC INSIDE assert_test
# ==========================================
# `assert_test` is what `deepeval test run` invokes. When a metric's score
# falls below its threshold the assertion fails — turning your custom
# metric into a CI/CD gate just like any built-in deepeval metric.

metric = LengthMetric(min_length=3, max_length=50, threshold=0.5)

test_case = LLMTestCase(
    input="Who is the Fourth Hokage?",
    actual_output="The Fourth Hokage is Minato Namikaze, the Yellow Flash of the Leaf.",
)

# `assert_test` only fails when the assertion is wrong. Pass a passing case
# to see the success path. Replace with a failing case to see the gate.
try:
    assert_test(test_case, [metric])
    print("assert_test: PASSED")
except AssertionError:
    print("assert_test: FAILED")