from basemodel import CustomOpenAI
from deepeval.synthesizer import Synthesizer

# ==========================================
# 1. MULTI-TURN FROM CONTEXTS
# ==========================================
# Same idea as the single-turn variant, but produces
# `ConversationalGolden` (with `scenario`, `turns`, and
# `expected_outcome`) instead of `Golden`.
#
# Multi-turn generation uses a
# `ConversationalStylingConfig` (set on the Synthesizer)
# to guide persona/scenario when no explicit seed
# goldens exist. For from-contexts that styling config
# defaults to a generic assistant persona unless
# overridden at Synthesizer construction time.

contexts = [
    [
        "All customers are eligible for a 30-day full refund at no extra cost.",
        "To request a refund, open the support page, enter the order number, and select 'Request refund'.",
        "Refunds are processed within five business days and appear on the original payment method.",
    ],
    [
        "To update the email on your account, sign in, go to 'Account settings', then 'Email'.",
        "You will receive a confirmation link at the new address; the change takes effect after you click it.",
        "To change your password, use the 'Forgot password' link on the sign-in page.",
    ],
]

# ==========================================
# 2. THE CALL
# ==========================================
# One mandatory parameter (`contexts`) and three
# optional ones:
#   - include_expected_outcome (default True)
#   - max_goldens_per_context  (default 2)
#   - source_files            (length must match `contexts`)

synthesizer = Synthesizer(model=CustomOpenAI())

conversational_goldens = synthesizer.generate_conversational_goldens_from_contexts(
    contexts=contexts,
    include_expected_outcome=True,
    max_goldens_per_context=2,
)

for i, golden in enumerate(conversational_goldens):
    print(f"--- conversational golden #{i} ---")
    print(f"scenario:        {golden.scenario}")
    print(f"expected_outcome:{golden.expected_outcome}")
    for turn in golden.turns:
        print(f"  [{turn.role}] {turn.content}")