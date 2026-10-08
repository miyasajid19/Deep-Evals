from basemodel import CustomOpenAI, OllamaEmbedding
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.simulator import ConversationSimulator
from deepeval.test_case import Turn
from typing import List

# ==========================================
# 1. GENERATE CONVERSATIONAL GOLDENS FROM DOCS
# ==========================================
synthesizer = Synthesizer(model=CustomOpenAI())

conversational_goldens = synthesizer.generate_conversational_goldens_from_docs(
    document_paths=["F:/Sajid_Miya_AI_Engineer_resume.pdf"],
    include_expected_outcome=True,
    max_goldens_per_context=2,
    context_construction_config=ContextConstructionConfig(
        embedder=OllamaEmbedding(),
        critic_model=CustomOpenAI(),
    ),
)

# ==========================================
# 2. SAVE GOLDENS
# ==========================================
synthesizer.save_as(file_type="csv", directory="./synthetic_data", file_name="my_dataset3")
synthesizer.save_as(file_type="json", directory="./synthetic_data", file_name="my_dataset3")
synthesizer.save_as(file_type="jsonl", directory="./synthetic_data", file_name="my_dataset3")

# ==========================================
# 3. DEFINE MODEL CALLBACK
# ==========================================
async def model_callback(input: str, turns: List[Turn], thread_id: str) -> Turn:
    # Replace with your actual LLM/chatbot logic
    answer = f"Response to: {input}"
    return Turn(role="assistant", content=answer)

# ==========================================
# 4. SIMULATE TURNS
# ==========================================
simulator = ConversationSimulator(
    model_callback=model_callback,
    simulator_model=CustomOpenAI(),  # <-- pass your custom model here
)

conversational_test_cases = simulator.simulate(
    conversational_goldens=conversational_goldens,
    max_user_simulations=10,
)

# ==========================================
# 5. INSPECT THE OUTPUT
# ==========================================
for i, test_case in enumerate(conversational_test_cases[:3]):
    print(f"--- conversational test case #{i} ---")
    print(f"scenario:         {test_case.scenario}")
    print(f"expected_outcome: {test_case.expected_outcome}")
    if test_case.turns:
        for turn in test_case.turns:
            print(f"  [{turn.role}] {turn.content}")
    else:
        print("  (no turns generated)")