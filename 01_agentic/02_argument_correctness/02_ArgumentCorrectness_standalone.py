from basemodel import CustomOpenAI
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from rich import print

# ==========================================
# 1. DEFINE THE METRIC
# ==========================================
# Standalone execution is useful for debugging or building a custom
# pipeline, but it does NOT include the testing reports, Confident AI
# integration, or speed / caching that `evaluate()` provides.

metric = ArgumentCorrectnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 2. BUILD A SINGLE LLMTestCase
# ==========================================
test_case = LLMTestCase(
    input="When did Naruto Uzumaki first learn the Rasengan?",
    actual_output=(
        "Naruto learned the Rasengan during the time-skip before Shippuden, "
        "training under Jiraiya using a water balloon."
    ),
    tools_called=[
        ToolCall(
            name="NarutoWikiSearch",
            description="Search the Naruto wiki for character or technique history.",
            input={"query": "Rasengan training", "character": "Naruto Uzumaki"},
        ),
        ToolCall(
            name="EpisodeLookup",
            description="Look up the episode where a character first uses a technique.",
            input={"character": "Naruto", "technique": "Rasengan"},
        ),
    ],
)

# ==========================================
# 3. MEASURE ON A SINGLE TEST CASE
# ==========================================
metric.measure(test_case)
print(f"Score:   {metric.score}")
print(f"Reason:  {metric.reason}")
print(f"Passed:  {metric.is_successful()}")