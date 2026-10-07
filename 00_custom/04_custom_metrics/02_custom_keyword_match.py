from __future__ import annotations
import re
from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric
from rich import print

# ==========================================
# 1. DEFINE A REGEX / KEYWORD CUSTOM METRIC
# ==========================================
# This metric checks that the actual output mentions a required set of
# keywords, plus an optional regex pattern. It's a deterministic, non-LLM
# gate useful for things like "must mention Minato" or "must include #."

class KeywordMatchMetric(BaseMetric):
    def __init__(
        self,
        required_keywords: list[str],
        regex_patterns: list[str] | None = None,
        threshold: float = 0.5,
    ):
        self.threshold = threshold
        self.required_keywords = [k.lower() for k in required_keywords]
        self.regex_patterns = regex_patterns or []
        # Compile once at init for efficiency.
        self._regexes = [re.compile(p) for p in self.regex_patterns]

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            text = (test_case.actual_output or "").lower()
            hits = sum(1 for k in self.required_keywords if k in text)
            keyword_score = hits / len(self.required_keywords) if self.required_keywords else 1.0

            regex_hits = sum(1 for r in self._regexes if r.search(test_case.actual_output or ""))
            regex_score = regex_hits / len(self._regexes) if self._regexes else 1.0

            # Combine: equal weight between keyword and regex components.
            self.score = (keyword_score + regex_score) / 2
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
        return "Keyword Match Metric"

# ==========================================
# 2. EXAMPLE USAGE
# ==========================================
metric = KeywordMatchMetric(
    required_keywords=["minato", "hokage"],
    regex_patterns=[r"yellow flash"],
    threshold=0.6,
)

test_cases = [
    LLMTestCase(
        input="Who is the Fourth Hokage?",
        # Hits "minato", "hokage", and the "yellow flash" regex → score 1.0
        actual_output="The Fourth Hokage is Minato Namikaze, the Yellow Flash.",
    ),
    LLMTestCase(
        input="Who is the Fourth Hokage?",
        # Hits "hokage" only, no "minato", no regex → score 0.5
        actual_output="The Fourth Hokage was the leader of the Hidden Leaf Village.",
    ),
    LLMTestCase(
        input="Who is the Fourth Hokage?",
        # Hits "minato" only → score 0.5
        actual_output="Minato Namikaze.",
    ),
]

for tc in test_cases:
    metric.measure(tc)
    print(f"Score:   {metric.score:.3f}")
    print(f"Passed:  {metric.is_successful()}")
    print("-" * 40)