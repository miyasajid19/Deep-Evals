from deepeval.tracing import observe, update_current_trace
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.test_case import LLMTestCase
from deepeval.metrics.community import AgentLoopDetectionMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# AgentLoopDetectionMetric is a fully DETERMINISTIC
# community metric — no LLM is involved, no API key,
# zero token cost. It analyzes the agent's full
# execution trace and returns a score in [0, 1].
#
# Three independent sub-signals combine into the score:
#   - Tool Call Repetition  (40%): same tool called
#     with identical args >= repetition_threshold
#     times.
#   - Reasoning Stagnation   (35%): consecutive LLM
#     outputs share >= similarity_threshold similarity.
#   - Call Graph Cycles     (25%): DFS finds a span
#     twice on the same root-to-leaf ancestry path
#     (true recursion).
#
# It runs in production without any LLM call. Score 1.0
# is clean; 0.0 is a critical loop.
#
# It only requires `input` and `actual_output` — no
# `tools_called` or `expected_tools`. Trace data is
# read from the agent trace set via `update_current_trace`
# inside an `@observe`d function.

# ==========================================
# 2. THE AGENT
# ==========================================
# A clean two-step agent: search, then compose. Each
# tool call uses a distinct query, so the trace shows
# no repetition and no cycles.

@observe()
def search_web(query: str) -> str:
    return f"Results for: {query}"

@observe()
def my_agent(input: str) -> str:
    result_1 = search_web(input)
    result_2 = search_web(f"summary of {input}")
    final = f"{result_1}; {result_2}"
    update_current_trace(input=input, output=final)
    return final

# ==========================================
# 3. THE DATASET + METRIC
# ==========================================

dataset = EvaluationDataset(goldens=[Golden(input="What is the weather in Paris?")])
loop_metric = AgentLoopDetectionMetric(threshold=0.5)

# ==========================================
# 4. RUN THE EVAL
# ==========================================
# `evals_iterator(metrics=[...])` runs the agent on
# each golden and evaluates the metric against the
# trace it captures.

for golden in dataset.evals_iterator(metrics=[loop_metric]):
    my_agent(golden.input)