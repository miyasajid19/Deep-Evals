from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE TWO AGENTS, ONE LEAN, ONE WASTEFUL
# ==========================================
# strict_mode collapses the 0–1 alignment score to a binary verdict:
#   - score 1 → every step in the trace was strictly required
#   - score 0 → anything wasteful (duplicates, retries, detours) tanks it
# It also overrides `threshold` to 1.0 internally. Useful for a hard
# pass/fail CI gate rather than a graded score.

@observe()
def lean_weather_agent(input: str):
    # `input` ignored — both agents answer the same canonical question.
    _ = input

    @observe()
    def get_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    return get_weather("Berlin")


@observe()
def wasteful_weather_agent(input: str):
    # Same answer, but with several redundant spans that strict mode
    # will flag: re-running the same tool, speculative lookups, logging.
    _ = input

    @observe()
    def get_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    @observe()
    def get_weather_again(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    @observe()
    def lookup_capital(country: str):
        return {"country": country, "capital": "Berlin"}

    @observe()
    def log_payload(payload: dict):
        return payload

    weather = get_weather("Berlin")
    _ = get_weather_again("Berlin")
    _ = lookup_capital("Germany")
    _ = log_payload(weather)
    return weather


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

strict_metric = StepEfficiencyMetric(
    threshold=0.5,
    strict_mode=True,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[strict_metric]):
    # Lean agent → should score 1
    lean_weather_agent(golden.input)
    # Wasteful agent → should score 0
    wasteful_weather_agent(golden.input)