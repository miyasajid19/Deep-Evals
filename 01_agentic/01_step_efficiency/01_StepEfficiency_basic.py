from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE AN AGENT WITH @observe SPANS
# ==========================================
# Step Efficiency is trace-only: it scores the LEANNESS of the agent's
# execution path, not the outcome. The metric extracts the task from the
# root span's input, then penalizes any span that wasn't strictly required
# to finish the task.
#
# This example uses two agents side-by-side: a lean one and a chatty one
# that re-calls tools it already has results from.

@observe()
def efficient_weather_agent(input: str):
    # `input` is intentionally ignored — both agents answer the same
    # canonical question. Surface it via `input` so Pylance is happy.
    _ = input

    @observe()
    def get_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    return get_weather("Berlin")

@observe()
def no_call_weather_agent(input: str):
    # `input` is intentionally ignored — both agents answer the same
    # canonical question. Surface it via `input` so Pylance is happy.
    _ = input

    return {"weather":None}


@observe()
def chatty_weather_agent(input: str):
    # Same answer as the efficient agent, but padded with redundant spans:
    # the agent re-fetches the weather, calls a tool it doesn't need, and
    # logs the same payload twice. Step Efficiency will penalize this.
    _ = input

    @observe()
    def get_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    @observe()
    def get_weather_again(city: str):
        # Re-running the same tool call is the kind of duplicate the
        # metric should flag as wasteful.
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    @observe()
    def lookup_capital(country: str):
        # Not needed to answer the user's question — agent is being
        # speculative.
        return {"country": country, "capital": "Berlin"}

    @observe()
    def log_payload(payload: dict):
        # Pure logging spans add no value to the answer.
        return payload

    weather = get_weather("Berlin")
    again = get_weather_again("Berlin")
    capital = lookup_capital("Germany")
    _ = log_payload(weather)

    return {"weather": weather, "again": again, "capital": capital}


# ==========================================
# 2. BUILD A DATASET
# ==========================================
# Same Golden drives both agents — Step Efficiency decides which path
# was leaner.

dataset = EvaluationDataset(
    goldens=[Golden(input="What's the current weather in Berlin?")]
)

# ==========================================
# 3. DEFINE THE METRIC (LLM-as-a-judge, default)
# ==========================================
# No `task=` is passed — Step Efficiency extracts the task from the root
# span's input automatically. The LLM then scores the alignment between
# the extracted task and the agent's execution steps.

step_efficiency = StepEfficiencyMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
# `evals_iterator` runs the agent once per golden, collects the trace, and
# measures the metric against it.

for golden in dataset.evals_iterator(metrics=[step_efficiency]):
    no_call_weather_agent(golden.input)
    efficient_weather_agent(golden.input)
    chatty_weather_agent(golden.input)