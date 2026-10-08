from deepeval.tracing import observe, update_current_trace
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics.community import AgentLoopDetectionMetric

# ==========================================
# 1. THE METRIC (verbose)
# ==========================================
# `verbose_mode=True` prints the per-signal verdicts
# before the final score — tool repetition count,
# consecutive-output similarity, and any back-edges in
# the call graph. The fastest way to debug "why is my
# loop score 0.25".

loop_metric = AgentLoopDetectionMetric(
    threshold=0.5,
    verbose_mode=True,
)

# ==========================================
# 2. A LOOPING AGENT
# ==========================================
# The agent calls search_web six times with the SAME
# argument — tool repetition will fire with
# max_repeats = 6 (>= 2 * repetition_threshold = 6).
# Reasoning stagnation will also fire if the model
# outputs identical responses each step.

@observe()
def search_web(query: str) -> str:
    return f"results: {query}"

@observe()
def looping_agent(input: str) -> str:
    out = ""
    for _ in range(6):
        out = search_web(input)  # identical args every iteration
    update_current_trace(input=input, output=out)
    return out

# ==========================================
# 3. THE DATASET
# ==========================================

dataset = EvaluationDataset(goldens=[Golden(input="What is the weather in Paris?")])

# ==========================================
# 4. RUN THE EVAL
# ==========================================

for golden in dataset.evals_iterator(metrics=[loop_metric]):
    looping_agent(golden.input)