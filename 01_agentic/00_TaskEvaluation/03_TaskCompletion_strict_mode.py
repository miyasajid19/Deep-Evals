from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE TWO AGENTS, ONE COMPLETE, ONE INCOMPLETE
# ==========================================
# strict_mode collapses the 0–1 alignment score to a binary verdict:
#   - score 1 → the outcome perfectly accomplishes the task
#   - score 0 → anything short of perfection
# It also overrides `threshold` to 1.0 internally. Useful when you want a
# hard pass/fail gate for a CI check rather than a graded score.

@observe()
def complete_weather_agent(input: str):
    # `input` is intentionally ignored — both agents answer the same
    # canonical "What's the weather in Berlin?" so the diff is purely
    # in the agent's behavior. Surface it via `input` so Pylance is happy.
    _ = input

    @observe()
    def get_current_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    @observe()
    def format_report(payload: dict):
        return (
            f"Weather in {payload['city']}: {payload['temp_c']}°C, "
            f"{payload['condition']}."
        )

    weather = get_current_weather("Berlin")
    return format_report(weather)


@observe()
def incomplete_weather_agent(input: str):
    _ = input

    @observe()
    def get_current_weather(city: str):
        return {"city": city}  # forgot the temp + condition

    return get_current_weather("Berlin")


# ==========================================
# 2. BUILD A DATASET
# ==========================================
# One Golden drives both agents — strict_mode decides which one passes.

dataset = EvaluationDataset(
    goldens=[
        Golden(input="What's the current weather in Berlin?"),
    ]
)

# ==========================================
# 3. DEFINE THE METRIC IN STRICT MODE
# ==========================================
# `threshold=0.5` is overridden to 1 internally by strict_mode. We still
# pass it for readability — strict_mode always wins.

strict_metric = TaskCompletionMetric(
    threshold=0.5,
    strict_mode=True,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# =========================================
for golden in dataset.evals_iterator(metrics=[strict_metric]):
    # Run the complete agent first → should score 1
    complete_weather_agent(golden.input)
    # Then run the incomplete agent → should score 0
    incomplete_weather_agent(golden.input)