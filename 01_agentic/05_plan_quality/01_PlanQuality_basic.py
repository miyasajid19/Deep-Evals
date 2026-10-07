from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanQualityMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE TWO AGENTS, ONE WITH A STRONG PLAN, ONE WITH A WEAK PLAN
# ==========================================
# Plan Quality is trace-only: it extracts the task AND a plan from the
# agent's reasoning, then judges ONLY the plan (not execution) for
# completeness, specificity, and quality.
#
# - strong_plan_agent:  complete, specific, well-ordered plan
# - weak_plan_agent:    vague one-liner plan that's missing key steps

@observe()
def strong_plan_agent(input: str):
    plan = (
        "Plan: 1) Look up the weather in Paris. 2) Look up the local time "
        "in Paris. 3) Format a single response that includes both pieces "
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
    weather = get_weather("Paris")
    t = get_local_time("Paris")
    return format_combined(weather, t)


@observe()
def weak_plan_agent(input: str):
    # Vague one-liner: doesn't say which tool to call or which city to
    # look up. Plan Quality should mark this as low-quality.
    plan = "Plan: do the thing."

    @observe()
    def think(plan_text: str):
        return plan_text

    @observe()
    def get_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    @observe()
    def get_local_time(city: str):
        return {"city": city, "time": "14:32"}

    think(plan)
    weather = get_weather("Paris")
    t = get_local_time("Paris")
    return {"weather": weather, "time": t}


# ==========================================
# 2. BUILD A DATASET
# ==========================================
# Same Golden drives both agents — Plan Quality decides which plan is
# better-suited to the task.

dataset = EvaluationDataset(
    goldens=[Golden(input="What's the weather and local time in Paris?")]
)

# ==========================================
# 3. DEFINE THE METRIC (LLM-as-a-judge, default)
# ==========================================
# No `task=` is passed — Plan Quality extracts the task from the root
# span's input AND the plan from the agent's reasoning spans, then
# rates the alignment between task and plan.

plan_quality = PlanQualityMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[plan_quality]):
    strong_plan_agent(golden.input)
    weak_plan_agent(golden.input)