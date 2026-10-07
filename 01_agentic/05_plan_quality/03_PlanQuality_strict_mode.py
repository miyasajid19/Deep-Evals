from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanQualityMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE TWO AGENTS, ONE WITH A GREAT PLAN, ONE WITH A BAD ONE
# ==========================================
# strict_mode collapses the 0–1 alignment score to a binary verdict:
#   - score 1 → the plan is excellent for the task
#   - score 0 → anything short of perfection
# It also overrides `threshold` to 1.0 internally. Useful for a hard
# pass/fail CI gate rather than a graded score.

@observe()
def great_plan_agent(input: str):
    plan = (
        "Plan: 1) Look up the weather in Berlin. 2) Look up the local time "
        "in Berlin. 3) Format a single response that includes both pieces "
        "of information. 4) Return the response."
    )

    @observe()
    def think(plan_text: str):
        return plan_text

    @observe()
    def get_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    @observe()
    def get_local_time(city: str):
        return {"city": city, "time": "14:32"}

    @observe()
    def format_combined(weather: dict, time: dict):
        return (
            f"Weather in {weather['city']}: {weather['temp_c']}°C, "
            f"{weather['condition']}. Local time: {time['time']}."
        )

    think(plan)
    weather = get_weather("Berlin")
    t = get_local_time("Berlin")
    return format_combined(weather, t)


@observe()
def weak_plan_agent(input: str):
    # Plan is missing the time step entirely and is vague about what
    # "handle it" means — strict_mode will fail it.
    plan = "Plan: handle it somehow."

    @observe()
    def think(plan_text: str):
        return plan_text

    @observe()
    def get_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    think(plan)
    return get_weather("Berlin")


# ==========================================
# 2. BUILD A DATASET
# ==========================================
dataset = EvaluationDataset(
    goldens=[Golden(input="What's the weather and local time in Berlin?")]
)

# ==========================================
# 3. DEFINE THE METRIC IN STRICT MODE
# ==========================================
# `threshold=0.5` is overridden to 1 internally by strict_mode. We still
# pass it for readability — strict_mode always wins.

strict_metric = PlanQualityMetric(
    threshold=0.5,
    strict_mode=True,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[strict_metric]):
    # Great-plan agent → should score 1
    great_plan_agent(golden.input)
    # Weak-plan agent → should score 0
    weak_plan_agent(golden.input)