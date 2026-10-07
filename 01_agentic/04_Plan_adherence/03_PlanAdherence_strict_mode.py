from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanAdherenceMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE TWO AGENTS, ONE ADHERENT, ONE DEVIANT
# ==========================================
# strict_mode collapses the 0–1 alignment score to a binary verdict:
#   - score 1 → execution perfectly followed the plan
#   - score 0 → any deviation tanks the run
# It also overrides `threshold` to 1.0 internally. Useful for a hard
# pass/fail CI gate rather than a graded score.

@observe()
def adherent_weather_agent(input: str):
    plan = (
        "Plan: 1) Look up the weather in Berlin. 2) Format the report. "
        "3) Return the report."
    )

    @observe()
    def think(plan_text: str):
        return plan_text

    @observe()
    def get_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    @observe()
    def format_report(payload: dict):
        return (
            f"Weather in {payload['city']}: {payload['temp_c']}°C, "
            f"{payload['condition']}."
        )

    think(plan)
    weather = get_weather("Berlin")
    return format_report(weather)


@observe()
def deviant_weather_agent(input: str):
    # Same plan, but adds a speculative lookup AND skips the format step.
    plan = (
        "Plan: 1) Look up the weather in Berlin. 2) Format the report. "
        "3) Return the report."
    )

    @observe()
    def think(plan_text: str):
        return plan_text

    @observe()
    def get_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    @observe()
    def lookup_capital(country: str):
        # Unplanned step.
        return {"country": country, "capital": "Berlin"}

    think(plan)
    weather = get_weather("Berlin")
    _ = lookup_capital("Germany")
    return weather  # skipped the planned format step


# ==========================================
# 2. BUILD A DATASET
# ==========================================
dataset = EvaluationDataset(
    goldens=[Golden(input="What's the current weather in Berlin?")]
)

# ==========================================
# 3. DEFINE THE METRIC IN STRICT MODE
# ==========================================
# `threshold=0.5` is overridden to 1 internally by strict_mode. We still
# pass it for readability — strict_mode always wins.

strict_metric = PlanAdherenceMetric(
    threshold=0.5,
    strict_mode=True,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[strict_metric]):
    # Adherent agent → should score 1
    adherent_weather_agent(golden.input)
    # Deviant agent → should score 0
    deviant_weather_agent(golden.input)