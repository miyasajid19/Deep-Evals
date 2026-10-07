from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE A MATH TUTOR AGENT
# ==========================================
# `system_one` mode asks Jev three questions about the trace and weights
# them into a single score — no LLM is called at all. It's the cheapest
# mode and the most decisive (no extraction step), but it needs the Jev
# API configured (JEV_API_KEY).

@observe()
def math_tutor_agent(input: str):
    # `input` ignored — the agent is hardcoded to solve a single canonical
    # quadratic. The point is to demonstrate the trace shape that Jev
    # receives (each span's name/type/inputs/outputs).
    _ = input

    @observe()
    def classify_problem(question: str):
        return {"question": question, "category": "quadratic equation"}

    @observe()
    def solve_quadratic(question: str):
        # Correct roots for x^2 - 5x + 6 = 0 are x = 2, x = 3
        return {"question": question, "roots": [2, 3], "method": "factoring"}

    @observe()
    def explain_solution(answer: dict):
        return (
            f"The equation factors as (x - 2)(x - 3) = 0, so the roots are "
            f"x = {answer['roots'][0]} and x = {answer['roots'][1]}."
        )

    problem = classify_problem("Solve x^2 - 5x + 6 = 0")
    answer = solve_quadratic(problem["question"])
    return explain_solution(answer)


# ==========================================
# 2. BUILD A DATASET
# ==========================================
dataset = EvaluationDataset(
    goldens=[Golden(input="Solve x^2 - 5x + 6 = 0 and explain your work.")]
)

# ==========================================
# 3. DEFINE THE METRIC IN SYSTEM_ONE MODE
# ==========================================
# Jev judges the whole metric in one request. It receives the trace (each
# span's name, type, inputs, outputs, tool calls) plus the task. Three
# questions are weighted into a single 0–1 score:
#   1. Every step was strictly necessary (Noul,  weight 2)
#   2. The path is the most direct, no detours/loops (Noul,  weight 1)
#   3. Overall efficiency rating (Score, weight 1)
# The reason lists each answer with its probability, and `metric.confidence`
# reports how decisive Jev was.

step_efficiency = StepEfficiencyMetric(
    threshold=0.7,
    eval_mode="system_one",
    system_one_model="jev-latest",
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[step_efficiency]):
    math_tutor_agent(golden.input)

    # Score on a `system_one` metric comes from Jev's answers, not the
    # LLM-chain equation — keep a metric on one eval mode when comparing
    # results across runs.