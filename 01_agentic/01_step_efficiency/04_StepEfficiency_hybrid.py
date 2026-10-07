from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE A RECIPE AGENT
# ==========================================
# In `hybrid` eval mode the LLM still extracts the task and execution
# steps from the trace, but the ALIGNMENT SCORE is produced by Jev
# (System One) instead of an LLM judge. Jev returns a rating from
# "Highly inefficient" to "Perfectly efficient" mapped onto 0–1; the
# metric's reason states Jev's score + confidence.

@observe()
def recipe_agent(user_query: str):
    # `user_query` ignored — the agent is hardcoded to produce a
    # carbonara recipe. The metric extracts the task from the root span's
    # input and judges whether the agent's path was efficient.
    _ = user_query

    @observe()
    def fetch_ingredients(dish: str):
        ingredients_for = {
            "carbonara": [
                "400g spaghetti",
                "200g pancetta",
                "4 eggs",
                "pecorino",
                "black pepper",
            ],
            "bolognese": ["400g tagliatelle", "500g beef mince", "soffritto"],
        }
        return ingredients_for.get(dish, [])

    @observe()
    def fetch_steps(dish: str):
        steps_for = {
            "carbonara": [
                "Boil salted water and drain the pasta al dente.",
                "Render pancetta until golden.",
                "Whisk eggs with pecorino and pepper.",
                "Toss hot pasta with pancetta off the heat, then with the egg mixture.",
                "Plate immediately with extra pecorino.",
            ],
        }
        return steps_for.get(dish, [])

    ingredients = fetch_ingredients("carbonara")
    steps = fetch_steps("carbonara")

    return {"dish": "Carbonara", "ingredients": ingredients, "steps": steps}


# ==========================================
# 2. BUILD A DATASET
# ==========================================
dataset = EvaluationDataset(
    goldens=[Golden(input="Give me a recipe for spaghetti carbonara.")]
)

# ==========================================
# 3. DEFINE THE METRIC IN HYBRID MODE
# ==========================================
# `eval_mode="hybrid"` means:
#   - LLM extracts the task and execution steps from the trace
#   - Jev (a System One model) rates the efficiency, not the LLM
#   - if the Jev call fails, the metric falls back to the LLM
# `system_one_model` is the Jev model; defaults to jev-latest. If the
# Jev endpoint isn't configured, this will raise — set the required env
# var (`JEV_API_KEY`) or switch to `eval_mode="llm"`.

step_efficiency = StepEfficiencyMetric(
    threshold=0.7,
    eval_mode="hybrid",
    model=CustomOpenAI(),
    system_one_model="jev-latest",
    include_reason=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[step_efficiency]):
    recipe_agent(golden.input)