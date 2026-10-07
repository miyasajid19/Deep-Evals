from basemodel import CustomOpenAI
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE THE METRIC WITH available_tools
# ==========================================
# When `available_tools` is passed, ToolCorrectnessMetric is HYBRID:
#   1. Deterministic score = correctly_used_tools / total_tools_called
#   2. LLM judges whether the chosen tools were the MOST OPTIMAL for
#      the input out of all available options
#   3. Final score = MIN(both)
# This is the only configuration where Tool Correctness uses an LLM.


# ==========================================
# 2. LIST THE TOOLS THE AGENT COULD HAVE USED
# ==========================================
available_tools = [
    ToolCall(name="GetWeather", description="Get the current weather for a city."),
    ToolCall(name="GetLocalTime", description="Get the current local time for a city."),
    ToolCall(name="GetFunFact", description="Get a fun fact about a city or topic."),
    ToolCall(name="FindRestaurant", description="Find a restaurant by cuisine and city."),
    ToolCall(name="TranslateText", description="Translate text between languages."),
]

metric = ToolCorrectnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    available_tools=available_tools,
)


# ==========================================
# 3. BUILD LLMTestCases
# ==========================================
test_cases = [
    # Optimal picks — every called tool is the best one for the input.
    LLMTestCase(
        input="What's the weather in Paris?",
        actual_output="It's 18°C and partly cloudy in Paris.",
        tools_called=[ToolCall(name="GetWeather")],
        expected_tools=[ToolCall(name="GetWeather")],
    ),
    # Suboptimal picks — the agent used a fun-fact tool where weather
    # was the obviously right choice. LLM should penalise; final score
    # = min(deterministic, LLM).
    LLMTestCase(
        input="What's the weather in Paris?",
        actual_output="Did you know Paris has a famous tower?",
        tools_called=[ToolCall(name="GetFunFact")],
        expected_tools=[ToolCall(name="GetWeather")],
    ),
    LLMTestCase(
        input="what is 10+10?",
        actual_output="10+10=20",
        tools_called=[ToolCall(name="Calculator")],
        expected_tools=[ToolCall(name="Calculator")],
    ),
]

# ==========================================
# 4. RUN END-TO-END EVALUATION
# ==========================================
evaluate(
    test_cases=test_cases,
    metrics=[metric],
    # Some deepeval versions accept `additional_kwargs` to pass the
    # available_tools list into the metric. If your version doesn't,
    # pass `available_tools=available_tools` directly when constructing
    # the metric above.
)