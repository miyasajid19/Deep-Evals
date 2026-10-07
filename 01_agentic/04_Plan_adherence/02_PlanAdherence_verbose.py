from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanAdherenceMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE A RESEARCH AGENT THAT STATES A PLAN
# ==========================================
# A multi-step agent: it plans out the steps, then executes them. Verbose
# mode is most useful for verifying that the metric is extracting the
# right plan and the right execution sequence from the trace.

@observe()
def research_agent(input: str):
    plan = (
        "Plan: 1) Search for the requested topic. 2) Summarise the top "
        "result. 3) Return the summary as the answer."
    )

    @observe()
    def think(plan_text: str):
        return plan_text

    @observe()
    def search(topic: str):
        return {
            "topic": topic,
            "results": [
                "The Hidden Leaf Village is the main setting of Naruto.",
                "It is led by the Hokage.",
            ],
        }

    @observe()
    def summarise(results: dict):
        # Take the first result as the summary.
        return results["results"][0]

    @observe()
    def answer(summary: str):
        return summary

    think(plan)
    results = search("Hidden Leaf Village")
    summary = summarise(results)
    return answer(summary)


# ==========================================
# 2. BUILD A DATASET
# ==========================================
dataset = EvaluationDataset(
    goldens=[Golden(input="Tell me about the Hidden Leaf Village in Naruto.")]
)

# ==========================================
# 3. DEFINE THE METRIC IN VERBOSE MODE
# ==========================================
# `verbose_mode=True` prints every intermediate step:
#   1. The extracted task string
#   2. The extracted plan string
#   3. The extracted execution steps
#   4. The alignment prompt sent to the judge
#   5. The judge's verdict + reason
# Use it when debugging — for production runs prefer the default.

plan_adherence = PlanAdherenceMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[plan_adherence]):
    research_agent(golden.input)