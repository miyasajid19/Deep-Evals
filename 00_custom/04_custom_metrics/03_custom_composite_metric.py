from typing import Optional
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    BaseMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from rich import print
from basemodel import CustomOpenAI
# ==========================================
# 1. DEFINE A COMPOSITE CUSTOM METRIC
# ==========================================
# Combine two deepeval metrics into one score. Composite metrics are
# useful when you rarely want to care about one without the other —
# e.g. an answer that isn't relevant is just as bad as an unfaithful one.
#
# `measure()` runs them sequentially; `a_measure()` runs them concurrently
# so `run_async=True` in `assert_test()` actually parallelises them.

class FaithfulRelevancyMetric(BaseMetric):
    def __init__(
        self,
        threshold: float = 0.5,
        evaluation_model: Optional[str] = None,
        include_reason: bool = True,
        async_mode: bool = True,
        strict_mode: bool = False,
    ):
        self.threshold = 1 if strict_mode else threshold
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.async_mode = async_mode
        self.strict_mode = strict_mode

    # ---------------- sync ----------------
    def measure(self, test_case: LLMTestCase) -> float:
        try:
            relevancy_metric, faithfulness_metric = self._init_metrics()
            relevancy_metric.measure(test_case)
            faithfulness_metric.measure(test_case)
            self._set_score_reason_success(relevancy_metric, faithfulness_metric)
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    # ---------------- async ----------------
    async def a_measure(self, test_case: LLMTestCase) -> float:
        try:
            relevancy_metric, faithfulness_metric = self._init_metrics()
            # Run both concurrently.
            await relevancy_metric.a_measure(test_case)
            await faithfulness_metric.a_measure(test_case)
            self._set_score_reason_success(relevancy_metric, faithfulness_metric)
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    # ---------------- housekeeping ----------------
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
        return "Composite Relevancy Faithfulness Metric"

    # ---------------- helpers ----------------
    def _init_metrics(self):
        relevancy_metric = AnswerRelevancyMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode,
        )
        faithfulness_metric = FaithfulnessMetric(
            threshold=self.threshold,
            model=self.evaluation_model,
            include_reason=self.include_reason,
            async_mode=self.async_mode,
            strict_mode=self.strict_mode,
        )
        return relevancy_metric, faithfulness_metric

    def _set_score_reason_success(
        self,
        relevancy_metric: BaseMetric,
        faithfulness_metric: BaseMetric,
    ):
        # Composite score: take the lower of the two so a weak link drags down.
        composite_score = min(relevancy_metric.score, faithfulness_metric.score)
        self.score = 0 if self.strict_mode and composite_score < self.threshold else composite_score

        if self.include_reason:
            self.reason = (
                f"Relevancy: {relevancy_metric.reason}\n"
                f"Faithfulness: {faithfulness_metric.reason}"
            )

        self.success = self.score >= self.threshold

# ==========================================
# 2. EXAMPLE USAGE
# ==========================================
metric = FaithfulRelevancyMetric(evaluation_model=CustomOpenAI(),threshold=0.7)

test_cases = [
    LLMTestCase(
        input="Who is the Fourth Hokage?",
        actual_output="The Fourth Hokage is Minato Namikaze, also known as the Yellow Flash.",
        retrieval_context=["Minato Namikaze is the Fourth Hokage of the Hidden Leaf Village."],
    ),
    LLMTestCase(
        input="Who is the Fourth Hokage?",
        actual_output="Sasuke Uchiha is the Fourth Hokage.",
        retrieval_context=["Minato Namikaze is the Fourth Hokage of the Hidden Leaf Village."],
    ),
]

for tc in test_cases:
    metric.measure(tc)
    print(f"Score:   {metric.score:.3f}")
    print(f"Passed:  {metric.is_successful()}")
    print("-" * 40)