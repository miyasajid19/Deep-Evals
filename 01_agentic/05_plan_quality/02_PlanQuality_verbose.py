from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import PlanQualityMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE A RESEARCH AGENT WITH A STATED PLAN
# ==========================================
# A multi-step agent: it states the plan, then carries it out. Verbose
# mode is most useful for verifying that the metric is extracting the
# right plan from the trace and reading it the way you expect.

@observe()
def research_agent(input: str):
    plan = (
        "Plan: 1) Search for the requested topic. 2) Read the top result. "
        "3) Summarise it. 4) Return the summary."
    )

    @observe()
    def think(plan_text: str):
        return plan_text

    @observe()
    def search(topic: str):
        return {
            "topic": topic,
            "results": [
                "Konohagakure (the Hidden Leaf Village) is the main setting of Naruto.",
                "It is led by the Hokage, the strongest ninja in the village.",
            ],
        }

    @observe()
    def read(results: dict):
        return results["results"][0]

    @observe()
    def summarise(text: str):
        return text

    @observe()
    def answer(summary: str):
        return summary

    think(plan)
    results = search("Hidden Leaf Village")
    top = read(results)
    summary = summarise(top)
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
#   3. The alignment prompt sent to the judge
#   4. The judge's verdict + reason
# Use it when debugging — for production runs prefer the default.

plan_quality = PlanQualityMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[plan_quality]):
    research_agent(golden.input)