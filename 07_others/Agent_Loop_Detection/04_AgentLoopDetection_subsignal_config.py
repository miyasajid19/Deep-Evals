from deepeval.metrics.community import AgentLoopDetectionMetric

# ==========================================
# 1. SUB-SIGNAL CONFIGURATION
# ==========================================
# AgentLoopDetectionMetric exposes three sub-signals
# you can toggle independently. Disabling a sub-signal
# removes its weight from the denominator so the score
# is never penalized for that dimension.
#
#   - check_tool_repetition    (default True, weight 40%)
#   - check_reasoning_stagnation (default True, weight 35%)
#   - check_call_graph_cycles  (default True, weight 25%)
#
# Plus three tuning knobs:
#   - repetition_threshold   (default 3) — how many
#     identical tool calls before flagging.
#   - similarity_threshold (default 0.85) — minimum
#     similarity between consecutive LLM outputs.
#   - strict_mode (default False)
#   - verbose_mode (default False)

# ==========================================
# 2. CONFIG A — only check tool repetition
# ==========================================
# Disable stagnation and call-graph cycle checks, and
# make the repetition threshold stricter (flag after
# 2 identical calls instead of 3).

only_repetition = AgentLoopDetectionMetric(
    threshold=0.5,
    check_tool_repetition=True,
    check_reasoning_stagnation=False,
    check_call_graph_cycles=False,
    repetition_threshold=2,  # stricter than the default 3
)

print("Config A — only tool repetition, threshold=2:")
print(f"  enabled signals: tool_repetition (weight 40%)")

# ==========================================
# 3. CONFIG B — only check true recursive cycles
# ==========================================
# Disable both repetition and stagnation. Only call
# graph cycles matter — useful when you want to catch
# literal recursion in the span tree, not sequential
# repetition of tool calls.

only_cycles = AgentLoopDetectionMetric(
    threshold=0.5,
    check_tool_repetition=False,
    check_reasoning_stagnation=False,
    check_call_graph_cycles=True,
)

print("Config B — only call graph cycles:")
print(f"  enabled signals: call_graph_cycles (weight 25%)")

# ==========================================
# 4. CONFIG C — all signals, custom similarity
# ==========================================
# Lower similarity threshold makes stagnation detection
# more sensitive (catches less-overlapping outputs).

sensitive_stagnation = AgentLoopDetectionMetric(
    threshold=0.5,
    check_tool_repetition=True,
    check_reasoning_stagnation=True,
    check_call_graph_cycles=True,
    similarity_threshold=0.5,  # catch more outputs as stagnating
)

print("Config C — all signals, lower similarity threshold:")
print(f"  similarity_threshold=0.5 (more sensitive)")