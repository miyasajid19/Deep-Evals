from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase, MCPServer, MCPToolCall
from deepeval.metrics import MultiTurnMCPUseMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three conversations of varying
# MCP-use quality. The denominator is total MCP
# interactions, so every noisy or redundant call is held
# to the same standard.

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

multi_turn_mcp_use = MultiTurnMCPUseMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. CASE A — perfect every turn
# ==========================================
# Two MCP interactions, both right tool + right args.

perfect_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Weather in Paris?"),
        Turn(
            role="assistant",
            content="It's 18°C in Paris.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "Paris"}, result={"temp_c": 18})
            ],
        ),
        Turn(role="user",      content="And London?"),
        Turn(
            role="assistant",
            content="It's 9°C in London.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "London"}, result={"temp_c": 9})
            ],
        ),
    ],
    mcp_servers=mcp_servers,
)

# ==========================================
# 3. CASE B — one wrong arg
# ==========================================
# Second interaction calls for Tokyo when the user asked
# about London. Mean of (1.0, ~0.0) ≈ 0.5.

one_wrong_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Weather in Paris?"),
        Turn(
            role="assistant",
            content="It's 18°C in Paris.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "Paris"}, result={"temp_c": 18})
            ],
        ),
        Turn(role="user",      content="And London?"),
        Turn(
            role="assistant",
            content="It's 14°C in Tokyo.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "Tokyo"}, result={"temp_c": 14})  # wrong
            ],
        ),
    ],
    mcp_servers=mcp_servers,
)

# ==========================================
# 4. CASE C — both interactions wrong args
# ==========================================
# Mean of (~0.0, ~0.0) ≈ 0.0.

all_wrong_case = ConversationalTestCase(
    turns=[
        Turn(role="user",      content="Weather in Paris?"),
        Turn(
            role="assistant",
            content="It's 9°C in London.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "London"}, result={"temp_c": 9})  # wrong
            ],
        ),
        Turn(role="user",      content="And London?"),
        Turn(
            role="assistant",
            content="It's 18°C in Paris.",
            mcp_tools_called=[
                MCPToolCall(name="get_current_weather", args={"city": "Paris"}, result={"temp_c": 18})  # wrong
            ],
        ),
    ],
    mcp_servers=mcp_servers,
)

# ==========================================
# 5. RUN THE METRIC ON ALL THREE CASES
# ==========================================
def run(label, case):
    multi_turn_mcp_use.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {multi_turn_mcp_use.score:.3f}")
    print(f"  Passed: {multi_turn_mcp_use.is_successful()}")
    print(f"  Reason: {multi_turn_mcp_use.reason}")

run("A — perfect every turn",    perfect_case)
run("B — one wrong arg",         one_wrong_case)
run("C — both wrong args",       all_wrong_case)