from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics.community import ToolPermissionMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# ToolPermissionMetric is a fully DETERMINISTIC
# community metric — it checks the tools your agent
# called against an allowlist (and/or a denylist). No
# LLM, no API key, zero token cost, well suited as a CI
# gate.
#
# It is different from ToolCorrectnessMetric, which
# checks tools called against tools EXPECTED for the
# task. ToolPermission enforces LEAST PRIVILEGE: it
# flags any tool call outside the granted policy,
# regardless of whether the task was completed.
#
# Required: `tools_called`. Score = number of
# authorized tool calls / total tool calls. If no
# tools were called, the score is 1.0 (no permission
# boundary could be violated).
#
# At least one of `allowed_tools` or `denied_tools`
# must be provided. A denial ALWAYS takes precedence
# over an allow.

# ==========================================
# 2. THE METRIC (allowlist)
# ==========================================
# Only `search_kb` and `reply_to_customer` are
# authorized. Any other tool is unauthorized.

metric = ToolPermissionMetric(
    allowed_tools=["search_kb", "reply_to_customer"],  # allowlist
    threshold=1.0,                                    # any unauthorized call fails
    include_reason=True,
)

# ==========================================
# 3. THE TEST CASE — authorized call
# ==========================================
# Agent called search_kb — on the allowlist. Expect
# score 1.0.

test_case = LLMTestCase(
    input="What is my refund status?",
    actual_output="Your refund is being processed.",
    tools_called=[ToolCall(name="search_kb")],
)

# ==========================================
# 4. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and
# writes a report (Confident AI if configured, else
# local cache).

evaluate(test_cases=[test_case], metrics=[metric])