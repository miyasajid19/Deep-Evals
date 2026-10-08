from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MCPUseMetric

# ==========================================
# 1. THE METRIC (hybrid mode)
# ==========================================
# Hybrid for MCPUse: the LLM still extracts the relevant
# alignment inputs (primitive calls, argument schemas).
# For each primitive use, Jev (a System One model) rates
# both halves on a FOUR-level scale mapped onto 0..1, with
# a short line stating Jev's score and confidence as
# each score's reason.
#
# The equation is unchanged. The LLM-written final reason
# is unchanged. If a Jev call fails, the metric silently
# defers that decision to the LLM.

mcp_servers = [
    MCPServer(
        server_name="weather",
        transport="stdio",
        available_tools=[
            {
                "name": "get_current_weather",
                "description": "Look up the current weather for a city.",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        ],
    )
]

hybrid_mcp_use = MCPUseMetric(
    threshold=0.5,
    model=CustomOpenAI(),       # LLM still writes the final reason
    include_reason=True,
    eval_mode="hybrid",         # Jev rates primitive + args on a 4-level scale
)

# ==========================================
# 2. THE TEST CASE
# ==========================================
test_case = LLMTestCase(
    input="What's the current weather in Paris?",
    actual_output="It's 18°C and partly cloudy in Paris right now.",
    mcp_servers=mcp_servers,
    mcp_tools_called=[
        MCPToolCall(
            name="get_current_weather",
            args={"city": "Paris"},
            result={"temp_c": 18, "condition": "Partly cloudy"},
        )
    ],
)

# ==========================================
# 3. RUN STANDALONE
# ==========================================
hybrid_mcp_use.measure(test_case)
print("---")
print(f"Score:      {hybrid_mcp_use.score}")
print(f"Confidence: {hybrid_mcp_use.confidence}")
print(f"Reason:     {hybrid_mcp_use.reason}")
print(f"Passed:     {hybrid_mcp_use.is_successful()}")