from basemodel import CustomOpenAI
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import Golden, ConversationalGolden
from deepeval.test_case import Turn

# ==========================================
# 1. WHY FROM-GOLDENS?
# ==========================================
# `generate_goldens_from_goldens` expands an existing
# dataset by generating more inputs that are similar in
# style/structure to your seed goldens. No documents or
# contexts are required.
#
# Two paths inside the method:
#   - If a seed golden has `context`, that context
#     grounds the new generation (same path as
#     from-contexts).
#   - Otherwise the synthesizer uses
#     `generate_goldens_from_scratch` with a StylingConfig
#     derived from the seed input (or one you provide
#     at Synthesizer construction time).
#
# `expected_output` is only generated for new goldens
# when the seed goldens carry `context` — that's how
# deepeval keeps the outputs grounded and not
# hallucinated.
#
# Rule of mixing: single-turn seeds only spawn single
# turn; multi-turn seeds only spawn multi-turn.

# ==========================================
# 2. SEED GOLDS (SINGLE-TURN)
# ==========================================
# Each seed Golden is `input` + (optionally) `context`.
# Provide a few diverse seeds so the expansion covers
# the topic surface you care about.

single_turn_seeds = [
    Golden(
        input="How long do refunds take?",
        context=[
            "All customers are eligible for a 30-day full refund at no extra cost.",
            "Refunds are processed within five business days and appear on the original payment method.",
        ],
    ),
    Golden(
        input="Can I change the email on my account?",
        context=[
            "To update the email on your account, sign in, go to 'Account settings', then 'Email'.",
            "You will receive a confirmation link at the new address; the change takes effect after you click it.",
        ],
    ),
]

# ==========================================
# 3. EXPAND SINGLE-TURN
# ==========================================
# One mandatory parameter (`goldens`) and two optional:
#   - max_goldens_per_golden  (default 2)
#   - include_expected_output (default True)

synthesizer = Synthesizer(model=CustomOpenAI())

expanded_single = synthesizer.generate_goldens_from_goldens(
    goldens=single_turn_seeds,
    max_goldens_per_golden=2,
    include_expected_output=True,
)

print(f"Expanded {len(single_turn_seeds)} seeds into {len(expanded_single)} single-turn goldens")
for golden in expanded_single:
    print(f"  - {golden.input}")

# ==========================================
# 4. SEED GOLDS (MULTI-TURN)
# ==========================================
# Each ConversationalGolden needs `scenario` and
# `turns`. `expected_outcome` is optional on the seed;
# it's not used by the expander, only `scenario` + `turns`
# seed the persona and conversation arc.

multi_turn_seeds = [
    ConversationalGolden(
        scenario="A customer asking about refund timelines and return shipping labels.",
        expected_outcome="The assistant explains the 30-day window, the five-business-day processing time, and how to use the prepaid return label.",
        turns=[
            Turn(role="user",      content="Hi, when will my refund land?"),
            Turn(role="assistant", content="Refunds are processed within five business days."),
            Turn(role="user",      content="Got it. Do I need to pay for return shipping?"),
            Turn(role="assistant", content="No — use the one in your order confirmation email."),
        ],
    ),
]

# ==========================================
# 5. EXPAND MULTI-TURN
# ==========================================
# Same shape, different goldens type. The synthesizer
# uses the seed `scenario` and `turns` to draft new
# `ConversationalGolden`s with fresh arcs.

expanded_multi = synthesizer.generate_conversational_goldens_from_goldens(
    goldens=multi_turn_seeds,
    max_goldens_per_golden=2,
    include_expected_outcome=True,
)

print(f"\nExpanded {len(multi_turn_seeds)} seeds into {len(expanded_multi)} multi-turn goldens")
for golden in expanded_multi:
    print(f"  - scenario: {golden.scenario}")
    print(f"    expected_outcome: {golden.expected_outcome}")