from deepeval.tracing import observe, update_current_trace
from deepeval.test_case import LLMTestCase
from deepeval.metrics.community import AgentLoopDetectionMetric

# ==========================================
# 1. THE METRIC (strict mode)
# ==========================================
# `strict_mode=True` collapses the score to binary:
#   - 1 if EVERY sub-signal reports 1.0 (no issue)
#   - 0 if any sub-signal drops below 1.0
# It also overrides `threshold` to 1, so
# `is_successful()` matches the binary score exactly.
#
# Use it as a CI gate: any hint of looping is a hard
# fail.

strict_loop_metric = AgentLoopDetectionMetric(
    threshold=0.5,             # ignored in strict mode
    strict_mode=True,
)

# ==========================================
# 2. CLEAN AGENT — score 1
# ==========================================
# Each tool call uses a distinct argument, so the
# repetition sub-signal stays at 1.0. Expect 1.

@observe()
def search(query: str) -> str:
    return f"results: {query}"

@observe()
def clean_agent(input: str) -> str:
    out = search(f"{input} detail 1")
    out = search(f"{input} detail 2")
    update_current_trace(input=input, output=out)
    return out

clean_case = LLMTestCase(
    input="What is the weather in Paris?",
    actual_output=clean_agent("What is the weather in Paris?"),
)

strict_loop_metric.measure(clean_case)
print("--- clean agent ---")
print(f"Score:   {strict_loop_metric.score}")
print(f"Reason:  {strict_loop_metric.reason}")
print(f"Passed:  {strict_loop_metric.is_successful()}")

# ==========================================
# 3. LOOPING AGENT — score 0
# ==========================================
# Six identical search_web calls fire the tool
# repetition sub-signal (6 >= 2 * 3 = 6). Strict mode
# drops the score to 0.

@observe()
def search_loop(query: str) -> str:
    return f"results: {query}"

@observe()
def looping_agent(input: str) -> str:
    out = ""
    for _ in range(6):
        out = search_loop(input)  # identical args every iteration
    update_current_trace(input=input, output=out)
    return out

loop_case = LLMTestCase(
    input="What is the weather in Paris?",
    actual_output=looping_agent("What is the weather in Paris?"),
)

strict_loop_metric.measure(loop_case)
print("--- looping agent ---")
print(f"Score:   {strict_loop_metric.score}")
print(f"Reason:  {strict_loop_metric.reason}")
print(f"Passed:  {strict_loop_metric.is_successful()}")