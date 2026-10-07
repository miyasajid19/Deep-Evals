from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE AN AGENT WITH @observe SPANS
# ==========================================
# Same trip planner as the basic example, but this time the root span's
# input ("Help me plan something fun.") is intentionally vague — we don't
# want Task Completion to infer the goal from it; we want to pin the goal
# explicitly via `task=` on the metric.

@observe()
def trip_planner_agent(input: str):
    # `input` is intentionally ignored here — the agent is hardcoded to
    # produce a 3-day Tokyo itinerary. The metric will judge that outcome
    # against the explicit `task` we set below, not the root span's input.
    _ = input
    destination = "Tokyo"
    days = 3

    @observe()
    def restaurant_finder(city: str):
        # `city` drives the span's input — shows up in the trace and
        # demonstrates the inner functions depend on the agent's params.
        city_restaurants = {
            "Tokyo": ["Sukiyabashi Jiro", "Ramen Nagi", "Ichiran Shibuya"],
            "Paris": ["Le Jules Verne", "Angelina Paris"],
        }
        return city_restaurants.get(city, [])

    @observe()
    def itinerary_generator(destination: str, days: int):
        # Both parameters are recorded as the span's input — proving that
        # inner spans surface their parameters in the trace.
        attractions = {
            "Tokyo": [
                "Senso-ji Temple",
                "Tokyo Skytree",
                "Meiji Shrine",
                "Shibuya Crossing",
                "Tsukiji Outer Market",
            ],
        }
        return attractions.get(destination, [])[:days]

    itinerary = itinerary_generator(destination, days)
    restaurants = restaurant_finder(destination)

    return {"destination": destination, "days": days, "itinerary": itinerary, "restaurants": restaurants}


# ==========================================
# 2. BUILD A DATASET
# ==========================================
# The Golden.input is the user-facing query that gets fed to the agent.
# It can be intentionally generic; the explicit `task` on the metric pins
# the evaluation against the real goal.

dataset = EvaluationDataset(
    goldens=[Golden(input="Help me plan something fun."),Golden(input="Plan a trip."),]
)

# ==========================================
# 3. DEFINE THE METRIC WITH AN EXPLICIT `task`
# ==========================================
# Supplying `task=` overrides the inferred task. Useful when:
#   - the root span's input is non-canonical (logging, IDs, etc.)
#   - you want one dataset row to be evaluated against multiple goals
#   - the agent's user-facing goal differs from its internal prompt

task_completion = TaskCompletionMetric(
    threshold=0.7,
    task=(
        "Plan a 3-day Tokyo itinerary that includes at least 3 attractions "
        "and 3 restaurant recommendations."
    ),
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# =========================================
for golden in dataset.evals_iterator(metrics=[task_completion]):
    trip_planner_agent(golden.input)