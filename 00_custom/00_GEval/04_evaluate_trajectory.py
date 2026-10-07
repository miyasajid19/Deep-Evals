from basemodel import CustomOpenAI
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe
from deepeval.metrics import GEval
from rich import print

# Trajectory GEval — no evaluation_params, judges the full trace
trajectory = GEval(
    model=CustomOpenAI(),
    name="Tool Use",
    criteria="Did the agent call the right tool with the right arguments, and does its final answer match what the tool returned?",
)

# Tool span — looks up a Naruto character's jutsu
@observe(type="tool")
def jutsu_lookup(character: str):
    jutsu_db = {
        "naruto": {"jutsu": "Rasengan", "rank": "A-rank"},
        "sasuke": {"jutsu": "Chidori", "rank": "A-rank"},
        "kakashi": {"jutsu": "Kamui", "rank": "S-rank"},
    }
    return jutsu_db.get(character.lower(), {"jutsu": "Unknown", "rank": "Unknown"})

# Tool span — looks up a character's village
@observe(type="tool")
def village_lookup(character: str):
    village_db = {
        "naruto": "Hidden Leaf Village (Konohagakure)",
        "sasuke": "Hidden Leaf Village (Konohagakure)",
        "gaara": "Hidden Sand Village (Sunagakure)",
    }
    return village_db.get(character.lower(), "Unknown Village")

# Agent span — orchestrates tool calls and produces final answer
@observe(type="agent")
def ninja_info_agent(input: str):
    # Simulated character extraction from input
    character = "naruto"
    if "sasuke" in input.lower():
        character = "sasuke"
    elif "kakashi" in input.lower():
        character = "kakashi"
    elif "gaara" in input.lower():
        character = "gaara"

    jutsu_info = jutsu_lookup(character)
    village_info = village_lookup(character)

    return (
        f"{character.capitalize()}'s signature jutsu is {jutsu_info['jutsu']} "
        f"(Rank: {jutsu_info['rank']}), and they are from {village_info}."
    )

# Dataset with Naruto-themed goldens
dataset = EvaluationDataset(goldens=[
    Golden(input="What is Naruto's jutsu and where is he from?"),
    Golden(input="Tell me about Sasuke's jutsu and village."),
    Golden(input="What jutsu does Kakashi use?"),
])

# Trajectory evaluation loop — metric judges the full trace
for golden in dataset.evals_iterator(metrics=[trajectory]):
    ninja_info_agent(golden.input)