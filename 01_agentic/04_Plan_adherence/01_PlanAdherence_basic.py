from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanAdherenceMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE TWO AGENTS, ONE ADHERES, ONE DEVIATES
# ==========================================
# Plan Adherence is trace-only: it extracts the task AND a plan from the
# agent's reasoning, then judges how faithfully execution followed the
# plan. The plan is surfaced in the trace by the agent's "thinking" or
# "reasoning" steps — here we capture that by stuffing the plan into a
# dedicated `think` span that Plan Adherence can extract from.
#
# - adherent_agent:  state the plan, then carry it out in order
# - deviant_agent:   state the plan, then take a step not in it

@observe()
def adherent_agent(input: str):
    # Surface the plan in the trace. The "Plan:" string is the signal
    # the LLM extractor looks for when reading the trace.
    plan = (
        "Plan: 1) Look up the weather in Paris. 2) Format the result. "
        "3) Return a correct result."
    )
    # `think` span carries the plan text into the trace.
    @observe()
    def think(plan_text: str):
        # The reasoning text becomes the agent's stated plan.
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

    think(plan)                                # state the plan
    weather = get_weather("Paris")              # step 1
    report = format_report(weather)             # step 2
    return report                              # step 3


@observe()
def deviant_agent(input: str):
    # Same plan, but the agent does an extra step (logging) that wasn't
    # in the plan. Plan Adherence should mark this as a deviation.
    plan = (
        "Plan: 1) Look up the weather in Paris. 2) Format the result. "
        "3) Return a friendly summary."
    )

    @observe()
    def think(plan_text: str):
        return plan_text

    @observe()
    def get_weather(city: str):
        return {"city": city, "temp_c": 18, "condition": "Partly cloudy"}

    @observe()
    def log_payload(payload: dict):
        # Step NOT in the plan — Plan Adherence should flag this.
        return payload

    @observe()
    def format_report(payload: dict):
        return (
            f"Weather in {payload['city']}: {payload['temp_c']}°C, "
            f"{payload['condition']}."
        )

    think(plan)
    weather = get_weather("Paris")
    _ = log_payload(weather)                   # unplanned step
    report = format_report(weather)
    return report


# ==========================================
# 2. BUILD A DATASET
# ==========================================
# One Golden drives both agents — Plan Adherence decides which run stuck
# to its stated plan.

dataset = EvaluationDataset(
    goldens=[Golden(input="What's the current weather in Paris?")]
)

# ==========================================
# 3. DEFINE THE METRIC (LLM-as-a-judge, default)
# ==========================================
# No `task=` is passed — Plan Adherence extracts the task from the root
# span's input AND the plan from the agent's reasoning spans, then rates
# the alignment between (task, plan) and the actual execution steps.

plan_adherence = PlanAdherenceMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[plan_adherence]):
    adherent_agent(golden.input)
    deviant_agent(golden.input)