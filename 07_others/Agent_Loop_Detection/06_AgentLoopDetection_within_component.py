from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.metrics.community import AgentLoopDetectionMetric

# ==========================================
# 1. WITHIN-COMPONENT USAGE
# ==========================================
# Like other metrics, AgentLoopDetectionMetric can be
# nested inside `@observe`d components via the
# `metrics=[...]` argument. The component-level
# evaluator pulls the metric, attaches the test case
# from `update_current_span`, and runs the metric
# against the inner trace.
#
# Useful when the loop metric should ONLY score the
# inner component (e.g. a planner node) and not the
# entire agent tree.

loop_metric = AgentLoopDetectionMetric(threshold=0.5)

@observe(metrics=[loop_metric])
def inner_agent_component(input: str) -> str:
    # Your agent logic here — kept minimal so the
    # trace shows a clean single-span execution.
    output = f"processed: {input}"
    test_case = LLMTestCase(input=input, actual_output=output)
    update_current_span(test_case=test_case)
    return output

@observe()
def outer_agent(input: str) -> str:
    return inner_agent_component(input)

# ==========================================
# 2. RUN THE EVAL
# ==========================================
# Calling outer_agent() runs inner_agent_component(),
# which captures the trace, attaches the test case,
# and runs the metric against the inner span only.

result = outer_agent("What is the weather in Paris?")
print(f"Output: {result}")
print(f"Score:  {loop_metric.score}")
print(f"Reason: {loop_metric.reason}")
print(f"Passed: {loop_metric.is_successful()}")