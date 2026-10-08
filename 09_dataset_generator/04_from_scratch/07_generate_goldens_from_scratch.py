from basemodel import CustomOpenAI
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    StylingConfig,
    ConversationalStylingConfig,
)

# ==========================================
# 1. WHY FROM-SCRATCH?
# ==========================================
# `generate_goldens_from_scratch` is the LEAST grounded
# generation mode — no documents, no contexts, no seed
# goldens. It exists for cases where your LLM app
# doesn't rely on RAG, or you want to stress-test on
# queries outside your knowledge base.
#
# Because there's nothing to ground the output, the
# synthesizer relies entirely on a `StylingConfig`
# (single-turn) or `ConversationalStylingConfig`
# (multi-turn) to know what kinds of inputs to invent.
# Construct that on the Synthesizer, not at call time.

# ==========================================
# 2. SINGLE-TURN: STYLINGCONFIG
# ==========================================
# StylingConfig fields:
#   scenario               — background situation for inputs
#   task                   — what the assistant should do
#   input_format           — shape of the user's input
#   expected_output_format — shape of the expected answer

styling_config = StylingConfig(
    scenario="Non-technical shoppers trying to track an online order or resolve a delivery issue.",
    task="Answering shipping/tracking-related queries using plain English.",
    input_format="Questions in English about order status, delivery timing, or shipping options.",
    expected_output_format="A short, plain-English answer that names the relevant policy (timing, location, etc.) and the next step the user can take.",
)

single_turn_synthesizer = Synthesizer(
    model=CustomOpenAI(),
    styling_config=styling_config,
)

# One mandatory parameter: `num_goldens`.
single_turn_goldens = single_turn_synthesizer.generate_goldens_from_scratch(
    num_goldens=10,
)

print(f"Generated {len(single_turn_goldens)} single-turn goldens from scratch")
for golden in single_turn_goldens[:5]:
    print(f"  - {golden.input}")

# ==========================================
# 3. MULTI-TURN: CONVERSATIONALSTYLINGCONFIG
# ==========================================
# ConversationalStylingConfig fields:
#   scenario_context       — overall context the convo lives in
#   conversational_task    — what the assistant is being tested on
#   participant_roles      — who shows up (e.g. "frustrated shopper")
#   scenario_format        — shape of the scenario text
#   expected_outcome_format — shape of the expected_outcome text

conversational_styling_config = ConversationalStylingConfig(
    scenario_context="An e-commerce support chat between a customer and a support assistant.",
    conversational_task="Resolving account and order issues (refund timing, email change, tracking).",
    participant_roles="Customer (frustrated, plain-English speaker) and a polite support assistant.",
)

multi_turn_synthesizer = Synthesizer(
    model=CustomOpenAI(),
    conversational_styling_config=conversational_styling_config,
)

multi_turn_goldens = multi_turn_synthesizer.generate_conversational_goldens_from_scratch(
    num_goldens=5,
)

print(f"\nGenerated {len(multi_turn_goldens)} multi-turn goldens from scratch")
for golden in multi_turn_goldens:
    print(f"  - scenario: {golden.scenario}")
    print(f"    expected_outcome: {golden.expected_outcome}")