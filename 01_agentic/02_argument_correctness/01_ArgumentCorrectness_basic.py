from basemodel import CustomOpenAI
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE THE METRIC (LLM-as-a-judge, default)
# ==========================================
# ArgumentCorrectnessMetric judges whether the ARGUMENTS (input
# parameters) for each tool call fit the input. It's referenceless — no
# expected tools / parameters needed. The LLM looks at each tool call
# individually and returns a 0–1 score = correct_arguments / total_calls.

metric = ArgumentCorrectnessMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. BUILD LLMTestCases WITH tools_called
# ==========================================
# Required: input, actual_output, tools_called. The LLM will judge whether
# the parameters in each ToolCall are appropriate for the input.

test_cases = [
    # Strong case: arguments are clearly aligned with the input.
    LLMTestCase(
        input="What's the weather in Paris right now?",
        actual_output="It's currently 18°C and partly cloudy in Paris.",
        tools_called=[
            ToolCall(
                name="GetWeather",
                description="Get the current weather for a given city.",
                input={"city": "Paris", "unit": "celsius"},
            ),
        ],
    ),
    # Weak case: tool is fine, but the argument is wrong city.
    LLMTestCase(
        input="What's the weather in Paris right now?",
        actual_output="It's currently 22°C and sunny in Berlin.",
        tools_called=[
            ToolCall(
                name="GetWeather",
                description="Get the current weather for a given city.",
                input={"city": "Berlin", "unit": "celsius"},
            ),
        ],
    ),
    # Multi-tool case: weather + restaurant lookup; both aligned.
    LLMTestCase(
        input="What's the weather in Tokyo and recommend a sushi restaurant?",
        actual_output=(
            "Tokyo is 25°C and sunny. For sushi, I'd recommend Sukiyabashi Jiro."
        ),
        tools_called=[
            ToolCall(
                name="GetWeather",
                description="Get the current weather for a given city.",
                input={"city": "Tokyo", "unit": "celsius"},
            ),
            ToolCall(
                name="FindRestaurant",
                description="Find a restaurant by cuisine and city.",
                input={"city": "Tokyo", "cuisine": "sushi"},
            ),
        ],
    ),
]

# ==========================================
# 3. RUN END-TO-END EVALUATION
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])