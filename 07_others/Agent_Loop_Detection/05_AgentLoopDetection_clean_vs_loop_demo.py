from deepeval.tracing import observe, update_current_trace
from deepeval.test_case import LLMTestCase
from deepeval.metrics.community import AgentLoopDetectionMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# Demo: a single metric instance scoring two agent
# traces at opposite ends of the loop spectrum.
#
# Score ranges:
#   - 1.0     clean execution
#   - 0.5–1.0 mild issues (recoverable)
#   - 0.0–0.5 severe looping (human review recommended)
#   - 0.0     critical loop (identical calls or true cycle)

loop_metric = AgentLoopDetectionMetric(threshold=0.5)

# ==========================================
# 2. AGENT A — clean run
# ==========================================
# Three tool calls, each with a distinct argument.
# Expect score 1.0.

@observe()
def fetch(url: str) -> str:
    return f"page: {url}"

@observe()
def clean_agent(input: str) -> str:
    p1 = fetch(f"https://api.example.com/{input}/detail")
    p2 = fetch(f"https://api.example.com/{input}/summary")
    p3 = fetch(f"https://api.example.com/{input}/related")
    out = f"{p1}; {p2}; {p3}"
    update_current_trace(input=input, output=out)
    return out

clean_case = LLMTestCase(
    input="weather Paris",
    actual_output=clean_agent("weather Paris"),
)

loop_metric.measure(clean_case)
print("--- A — clean agent (distinct tool args) ---")
print(f"  Score:    {loop_metric.score}")
print(f"  Breakdown: {loop_metric.score_breakdown}")
print(f"  Reason:   {loop_metric.reason}")
print(f"  Passed:   {loop_metric.is_successful()}")

# ==========================================
# 3. AGENT B — looping agent
# ==========================================
# Six identical tool calls fire tool repetition
# (6 >= 2 * 3 = 6). Expect a low score.

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
    input="weather Paris",
    actual_output=looping_agent("weather Paris"),
)

loop_metric.measure(loop_case)
print("--- B — looping agent (6 identical calls) ---")
print(f"  Score:    {loop_metric.score}")
print(f"  Breakdown: {loop_metric.score_breakdown}")
print(f"  Reason:   {loop_metric.reason}")
print(f"  Passed:   {loop_metric.is_successful()}")