from basemodel import CustomOpenAI
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import StepEfficiencyMetric
from deepeval.tracing import observe

# ==========================================
# 1. DEFINE A BOOK-RECOMMENDER AGENT
# ==========================================
# A small agent that looks up candidates, ranks them, and returns the
# top picks. Verbose mode is most useful when you want to see WHICH
# spans Step Efficiency flagged as wasteful.

@observe()
def book_recommender_agent(input: str):
    # Pull the genre out of the request. In a real agent this would be
    # an LLM call; the example uses keyword matching to keep the trace
    # deterministic.
    genre = "fantasy" if "fantasy" in input.lower() else "sci-fi"

    @observe()
    def lookup_candidates(genre: str):
        catalog = {
            "fantasy": [
                {"title": "The Name of the Wind", "author": "Patrick Rothfuss"},
                {"title": "Mistborn", "author": "Brandon Sanderson"},
                {"title": "The Way of Kings", "author": "Brandon Sanderson"},
            ],
            "sci-fi": [
                {"title": "Project Hail Mary", "author": "Andy Weir"},
                {"title": "The Three-Body Problem", "author": "Cixin Liu"},
            ],
        }
        return catalog.get(genre, [])

    @observe()
    def rank_top_picks(candidates: list, top_n: int):
        return candidates[:top_n]

    candidates = lookup_candidates(genre)
    picks = rank_top_picks(candidates, top_n=2)
    return {"genre": genre, "picks": picks}


# ==========================================
# 2. BUILD A DATASET
# ==========================================
dataset = EvaluationDataset(
    goldens=[
        Golden(input="Recommend a fantasy novel I might enjoy."),
        Golden(input="Recommend a sci-fi novel I might enjoy."),
    ]
)

# ==========================================
# 3. DEFINE THE METRIC IN VERBOSE MODE
# ==========================================
# `verbose_mode=True` prints every intermediate step:
#   1. The extracted task string
#   2. The extracted execution steps from the trace
#   3. The alignment prompt sent to the judge
#   4. The judge's verdict + reason
# Use it when debugging which spans get flagged as wasteful.

step_efficiency = StepEfficiencyMetric(
    threshold=0.7,
    model=CustomOpenAI(),
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 4. RUN THE EVALS ITERATOR
# ==========================================
for golden in dataset.evals_iterator(metrics=[step_efficiency]):
    book_recommender_agent(golden.input)