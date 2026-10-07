from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanQualityMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE A MATH TUTOR AGENT WITH A STATED PLAN
# ==========================================
# `system_one` mode asks Jev three choice questions about the trace and
# weights them into a single score — no LLM is called at all. Each
# question has a "The agent states no plan" option that is not applicable,
# so an agent with no plan scores 1. The questions are:
#   1. Would the plan, if carried out, fully accomplish the task? (weight 2)
#   2. Is every step specific, necessary, and in logical order?    (weight 1)
#   3. How good is the plan overall?                              (weight 1)
# The reason lists each answer with its probability, and `metric.confidence`
# reports how decisive Jev was.

@observe()
def math_tutor_agent(input: str):
    plan = (
        "Plan: 1) Identify the equation as a quadratic. 2) Solve for "
        "the roots by factoring. 3) Verify the roots by substitution. "
        "4) Explain the solution in plain English."
    )

    @observe()
    def think(plan_text: str):
        return plan_text

    @observe()
    def classify_problem(question: str):
        return {"question": question, "category": "quadratic equation"}

    @observe()
    def solve_quadratic(question: str):
        # Correct roots for x^2 - 5x + 6 = 0 are x = 2, x = 3
        return {"question": question, "roots": [2, 3], "method": "factoring"}

    @observe()
    def verify_roots(answer: dict):
        roots = answer["roots"]
        return {
            "roots": roots,
            "valid": all(2 + r - 6 == 0 for r in roots) is False,  # illustrative
        }

    @observe()
    def explain_solution(answer: dict):
        return (
            f"The equation factors as (x - 2)(x - 3) = 0, so the roots are "
            f"x = {answer['roots'][0]} and x = {answer['roots'][1]}."
        )

    think(plan)
    problem = classify_problem("Solve x^2 - 5x + 6 = 0")
    answer = solve_quadratic(problem["question"])
    _ = verify_roots(answer)
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
plan_quality = PlanQualityMetric(
    threshold=0.7,
    eval_mode="system_one",
    system_one_model="jev-latest",
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[plan_quality]):
    math_tutor_agent(golden.input)

    # Score on a `system_one` metric comes from Jev's answers, not the
    # LLM-chain equation — keep a metric on one eval mode when comparing
    # results across runs.