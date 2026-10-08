from basemodel import CustomOpenAI
from deepeval.classifiers import ToneAdherenceClassifier
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `ToneAdherenceClassifier` also accepts a
# `ConversationalTestCase`. It reads every turn along with
# the scenario and expected outcome, and chooses ONE label
# for the conversation as a whole. The default multi-turn
# template explicitly tells the judge to consider the
# conversation as a whole, not just the last turn, unless
# a label description says otherwise.
#
# Multi-turn is the natural setting for tone drift: the
# first reply is on script, the second drifts, the fourth
# is a five-paragraph policy explanation that the user
# never asked for. Per-turn classification would average a
# warm opener with a cold wind-down and report a mixed
# verdict; conversation-level classification catches the
# drift.

classifier = ToneAdherenceClassifier(
    tone="warm, plain-spoken, and under three sentences",
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE CONVERSATIONAL TEST CASES
# ==========================================
# `expected_labels` is a dict keyed by `classifier.name`.
# For `ToneAdherenceClassifier`, that key is fixed — the
# built-in `name` DeepEval assigns to this classifier —
# so you set the expected label as a string the judge
# will return.
#
# Two conversations against the same configured voice:
#
#   1) "on_tone" — the assistant stays warm, plain-spoken,
#      and under three sentences across the whole
#      exchange. The reply even acknowledges the complaint
#      concretely without falling back on policy boilerplate.
#   2) "off_tone" — the assistant opens on script but
#      slips into legalistic boilerplate by the third
#      reply. The conversation as a whole misses the
#      configured voice, even though some turns are fine in
#      isolation.
#
# `scenario` and `expected_outcome` are also read by the
# judge when present; populate them to give the judge
# more context about what the conversation was trying
# to do.

conversation_on_tone = ConversationalTestCase(
    scenario=(
        "A user reports a missing package; the assistant is "
        "configured to be warm, plain-spoken, and under "
        "three sentences."
    ),
    expected_outcome=(
        "Across every turn the assistant stays warm and "
        "brief, and resolves the complaint concretely."
    ),
    turns=[
        Turn(role="user", content="My package hasn't arrived."),
        Turn(
            role="assistant",
            content=(
                "Sorry about that! I've checked and it's out "
                "for delivery today. I'll keep an eye on it "
                "for you."
            ),
        ),
        Turn(role="user", content="It's still not here."),
        Turn(
            role="assistant",
            content=(
                "Apologies — I've flagged it for follow-up. "
                "If it's not with you by tomorrow, ping me "
                "and I'll escalate."
            ),
        ),
        Turn(role="user", content="Thanks."),
        Turn(
            role="assistant",
            content=(
                "Anytime — happy to help."
            ),
        ),
    ],
    expected_labels={classifier.name: "on_tone"},
)

conversation_off_tone = ConversationalTestCase(
    scenario=(
        "A user reports a missing package; the assistant is "
        "configured to be warm, plain-spoken, and under "
        "three sentences."
    ),
    expected_outcome=(
        "The assistant opens on script but drifts into "
        "legalistic boilerplate and policy references by "
        "the third reply."
    ),
    turns=[
        Turn(role="user", content="My package hasn't arrived."),
        Turn(
            role="assistant",
            content=(
                "Sorry about that! I'll look into it right "
                "now."
            ),
        ),
        Turn(role="user", content="It's been three days already."),
        Turn(
            role="assistant",
            content=(
                "Dear valued customer, pursuant to Section "
                "4.2 of our Customer Service Agreement, "
                "consignments may experience transit delays "
                "due to factors outside the carrier's direct "
                "control. We respectfully request your "
                "patience while our logistics partners "
                "complete the scheduled delivery cycle."
            ),
        ),
        Turn(role="user", content="Can I just get a refund?"),
        Turn(
            role="assistant",
            content=(
                "Should the package fail to arrive within "
                "five (5) business days from the date of "
                "this correspondence, kindly re-engage via "
                "the support channel and reference ticket "
                "#49281. Refunds are processed only after "
                "the investigation window closes and require "
                "supervisor approval."
            ),
        ),
    ],
    expected_labels={classifier.name: "off_tone"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# Both conversations use the same configured voice — only
# the assistant's reply style changes. The judge applies
# the same rubric to both, so the conversation-level
# tone-adherence rate reflects drift over the whole
# exchange, not a per-turn average.

evaluate(
    test_cases=[conversation_on_tone, conversation_off_tone],
    classifiers=[classifier],
)