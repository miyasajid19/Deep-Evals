from __future__ import annotations
from deepeval.test_case import ConversationalTestCase, Turn
from deepeval.metrics import BaseConversationalMetric
from rich import print

# ==========================================
# 1. DEFINE A MULTI-TURN CUSTOM METRIC
# ==========================================
# Multi-turn metrics inherit from BaseConversationalMetric and accept a
# ConversationalTestCase in measure()/a_measure(). This one checks every
# assistant turn for polite, on-topic language — no LLM, just heuristics.

class PolitenessTurnMetric(BaseConversationalMetric):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def measure(self, test_case: ConversationalTestCase) -> float:
        try:
            assistant_turns = [t for t in test_case.turns if t.role == "assistant"]
            if not assistant_turns:
                self.score = 0.0
                self.success = False
                return self.score

            scores = [self._score_turn(t.content or "") for t in assistant_turns]
            self.score = sum(scores) / len(scores)
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            self.error = str(e)
            raise

    async def a_measure(self, test_case: ConversationalTestCase) -> float:
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
        return "Politeness Turn Metric"

    # ---------------- heuristic ----------------
    @staticmethod
    def _score_turn(content: str) -> float:
        text = content.lower()
        rude_tokens = ["stupid", "idiot", "shut up", "dumb"]
        polite_tokens = ["please", "thank you", "happy to help", "of course"]

        if any(tok in text for tok in rude_tokens):
            return 0.0
        return 1.0 if any(tok in text for tok in polite_tokens) else 0.5

# ==========================================
# 2. EXAMPLE USAGE
# ==========================================
metric = PolitenessTurnMetric(threshold=0.5)

test_cases = [
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="Hi, can you help me?"),
            Turn(role="assistant", content="Of course! I'd be happy to help you. What do you need?"),
            Turn(role="user", content="Reset my password."),
            Turn(role="assistant", content="Sure, please share your email and I'll send a reset link."),
            Turn(role="user", content="Thanks!"),
            Turn(role="assistant", content="You're welcome!"),
        ]
    ),
    ConversationalTestCase(
        turns=[
            Turn(role="user", content="Hi, can you help me?"),
            Turn(role="assistant", content="What do you want?"),
            Turn(role="user", content="Reset my password."),
            Turn(role="assistant", content="Figure it out yourself."),
        ]
    ),
]

for tc in test_cases:
    metric.measure(tc)
    print(f"Score:   {metric.score:.3f}")
    print(f"Passed:  {metric.is_successful()}")
    print("-" * 40)