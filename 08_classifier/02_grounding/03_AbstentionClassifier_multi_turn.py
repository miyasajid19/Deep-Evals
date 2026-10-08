from basemodel import CustomOpenAI
from deepeval.classifiers import AbstentionClassifier
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `AbstentionClassifier` works on multi-turn test cases as
# well as single-turn ones. The judge considers the whole
# conversation — context, turns, and the assistant's final
# reply — when deciding whether the assistant answered,
# abstained, or fabricated. Its `name` and label set are
# fixed (the same three labels: "abstained", "answered",
# "fabricated"), so abstention rates are comparable across
# single-turn and multi-turn datasets in the same run.
#
# For multi-turn test cases the judge reads `scenario` (the
# user's goal), `context` (the retrieval context the
# application had available), and the full turn list, then
# picks one label for the conversation as a whole. The
# label is a property of the *exchange*, not any individual
# reply — a long conversation can end with the assistant
# either delivering the answer from context, declining
# honestly, or fabricating one.
#
# Optional constructor knobs (defaults shown):
#   - model                  — evaluation LLM (defaults to
#                              DeepEval's default GPT model).
#                              Pass a `DeepEvalBaseLLM` to use
#                              your own model, e.g. the
#                              `CustomOpenAI` defined in
#                              `basemodel.py` for this repo.
#   - include_reason=True    — when True, the judge also
#                              returns a short reason explaining
#                              the chosen label.
#   - allow_none=False       — when False (the default), the
#                              judge always picks the closest
#                              label. Set True to let the judge
#                              abstain (`label=None` with a
#                              reason) when none fit.
#   - async_mode=True        — runs `classify()` concurrently
#                              across test cases.
#   - classification_template — override the judge prompt.
#   - eval_mode              — "llm" (default) or "system_one"
#                              (Jev-as-a-judge, requires
#                              `TYPESAFE_API_KEY`).

classifier = AbstentionClassifier(
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE CONVERSATIONAL TEST CASES
# ==========================================
# `ConversationalTestCase` carries the conversation as a
# list of `Turn`s plus a `context` field that holds the
# retrieval context the application actually had. Without
# `context`, the classifier cannot separate "abstained"
# from "fabricated" — it has no way to know the answer was
# missing.
#
# We exercise all three label values across three realistic
# multi-turn flows against the same policy context:
#
#   - "answered"   — a user asks about standard-plan
#                    refunds; the assistant confirms the
#                    30-day window from the policy context.
#                    The context supports it, the response
#                    gives it.
#   - "abstained"  — a user asks about enterprise-plan
#                    refunds; the assistant honestly says
#                    the policy documents don't cover
#                    enterprise plans and redirects to the
#                    account team. The context doesn't have
#                    it, and the response says so.
#   - "fabricated" — a user asks about enterprise-plan
#                    refunds across two turns; the assistant
#                    invents a "60 days via your account
#                    manager" policy. The context doesn't
#                    have it, but the response makes one up.
#
# `scenario` states the goal; `expected_outcome` (optional)
# spells out the substantive ending the metric is testing
# for.

test_case_answered = ConversationalTestCase(
    scenario="User wants to know the refund window for standard plans.",
    expected_outcome=(
        "The user gets the standard-plan refund window "
        "from the policy context."
    ),
    context=[
        "Standard plans can be refunded within 30 days of "
        "purchase. Enterprise plans are handled by the "
        "account team and are not covered by this policy."
    ],
    turns=[
        Turn(
            role="user",
            content="Hi — I bought a standard plan last week. Can I still get a refund?",
        ),
        Turn(
            role="assistant",
            content=(
                "Yes. Standard plans can be refunded within "
                "30 days of purchase. Do you want me to "
                "start the refund?"
            ),
        ),
        Turn(role="user", content="Yes, please."),
        Turn(
            role="assistant",
            content=(
                "Done. I've submitted your refund. You'll "
                "see it back on your card within 5–7 days."
            ),
        ),
    ],
    expected_labels={classifier.name: "answered"},
)

test_case_abstained = ConversationalTestCase(
    scenario="User asks about the enterprise-plan refund window.",
    expected_outcome=(
        "The assistant declines honestly because the policy "
        "documents do not cover enterprise plans."
    ),
    context=[
        "Standard plans can be refunded within 30 days of "
        "purchase. Enterprise plans are handled by the "
        "account team and are not covered by this policy."
    ],
    turns=[
        Turn(
            role="user",
            content="Our company is on an enterprise plan. What's the refund window?",
        ),
        Turn(
            role="assistant",
            content=(
                "Our public policy documents only cover "
                "standard plans (30 days). Enterprise "
                "refunds are handled by your account team — "
                "would you like me to put you in touch?"
            ),
        ),
        Turn(role="user", content="Yes, please."),
        Turn(
            role="assistant",
            content=(
                "I'll have your account manager reach out "
                "by the end of the day."
            ),
        ),
    ],
    expected_labels={classifier.name: "abstained"},
)

test_case_fabricated = ConversationalTestCase(
    scenario="User asks about the enterprise-plan refund window.",
    expected_outcome=(
        "The assistant invents a refund window rather than "
        "admitting the policy documents do not cover it."
    ),
    context=[
        "Standard plans can be refunded within 30 days of "
        "purchase. Enterprise plans are handled by the "
        "account team and are not covered by this policy."
    ],
    turns=[
        Turn(
            role="user",
            content="Our company is on an enterprise plan. What's the refund window?",
        ),
        Turn(
            role="assistant",
            content=(
                "Enterprise plans can be refunded within 60 "
                "days. Just go through your account manager "
                "and they'll process it."
            ),
        ),
        Turn(
            role="user",
            content="Great. Can you start that for me?",
        ),
        Turn(
            role="assistant",
            content=(
                "I've flagged your account for a 60-day "
                "enterprise refund."
            ),
        ),
    ],
    expected_labels={classifier.name: "fabricated"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# All three labels are exercised against the same context
# in one run, so the answer rate, the honest-abstain rate,
# and the fabrication rate come from the same judge and
# the same context scale. That's the value of testing
# abstention through a fixed label set: drift in any one
# direction — the bot silently fabricating across a long
# exchange, or over-declining when the context actually
# has the answer — becomes a measurable regression rather
# than a vibes-based review.

evaluate(
    test_cases=[
        test_case_answered,
        test_case_abstained,
        test_case_fabricated,
    ],
    classifiers=[classifier],
)