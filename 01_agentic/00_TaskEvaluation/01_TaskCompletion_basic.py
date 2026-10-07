from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE AN AGENT WITH @observe SPANS
# ==========================================
# Task Completion scores the OUTCOME of the whole trace, so the agent must
# be instrumented. Every nested @observe call becomes a span in the trace;
# the metric reads the root span's input/output when inferring the task and
# outcome (unless you override with `task=...`).

@observe()
def trip_planner_agent(input: str):
    # In a real agent this would parse `input` via an LLM; for the example
    # we use a tiny heuristic so the trace stays deterministic.
    destination = "Paris" if "Paris" in input else "Tokyo"
    days = 2 if "2-day" in input else 3

    @observe()
    def restaurant_finder(city: str):
        # `city` drives the lookup so the span captures the actual context
        # the agent used, not a hardcoded constant.
        city_restaurants = {
            "Paris": ["Le Jules Verne", "Angelina Paris", "Septime"],
            "Tokyo": ["Sukiyabashi Jiro", "Ramen Nagi"],
        }
        return city_restaurants.get(city, [])

    @observe()
    def itinerary_generator(destination: str, days: int):
        # Both parameters are inputs to the span — they show up in the
        # trace's input field and demonstrate parameter propagation.
        attractions = {
            "Paris": ["Eiffel Tower", "Louvre Museum", "Montmartre"],
            "Tokyo": ["Senso-ji", "Tokyo Skytree", "Meiji Shrine"],
        }
        return {
            "destination": destination,
            "days": days,
            "attractions": attractions.get(destination, [])[:days],
        }

    itinerary = itinerary_generator(destination, days)
    restaurants = restaurant_finder(destination)

    return {"itinerary": itinerary, "restaurants": restaurants}


# ==========================================
# 2. BUILD A DATASET
# ==========================================
# Each Golden is a separate agent invocation that produces its own trace.

dataset = EvaluationDataset(
    goldens=[
        Golden(input="Plan a 2-day trip to Paris."),
        Golden(input="Plan a 3-day trip to Tokyo."),
    ]
)

# ==========================================
# 3. DEFINE THE METRIC (LLM-as-a-judge, default)
# ==========================================
# No `task=` is passed, so Task Completion extracts the task from the root
# span's input. The LLM then aligns that extracted task against the
# extracted outcome and emits a 0–1 score plus reason.

task_completion = TaskCompletionMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
# `evals_iterator` runs the agent once per golden, collects the trace, and
# measures the metric against it. Yielded goldens are for further inspection
# (e.g. logging); the metric results are printed at the end.

for golden in dataset.evals_iterator(metrics=[task_completion]):
    trip_planner_agent(golden.input)