from basemodel import CustomOpenAI
from deepeval.classifiers import RequiredDisclosureClassifier
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `RequiredDisclosureClassifier` also accepts a
# `ConversationalTestCase`. It reads every turn along with
# the scenario and expected outcome, and chooses ONE label
# for the conversation as a whole. The default multi-turn
# template explicitly tells the judge to consider the
# conversation as a whole, not just the last turn, unless
# a label description says otherwise.
#
# Multi-turn is the natural setting for disclosure drift:
# the first reply is fully compliant, the second forgets
# one element, the third forgets both. Per-turn
# classification would average the runs and report a
# mixed verdict; conversation-level classification catches
# whether the assistant as a whole upheld the policy
# throughout the exchange.

classifier = RequiredDisclosureClassifier(
    disclosures=[
        "a statement that this is not financial advice",
        "a recommendation to consult a licensed advisor",
    ],
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE CONVERSATIONAL TEST CASES
# ==========================================
# `expected_labels` is a dict keyed by `classifier.name`.
# For `RequiredDisclosureClassifier`, that key is fixed —
# the built-in `name` DeepEval assigns to this classifier —
# so you set the expected label as a string the judge
# will return.
#
# Three conversations against the same disclosure
# checklist:
#
#   1) "present" — across the whole exchange the assistant
#      includes both required elements in every reply that
#      gives substantive guidance. The compliant run.
#   2) "partial" — the first reply is fully compliant, the
#      second drops the "consult a licensed advisor"
#      line, the third drops both elements. As a whole the
#      conversation only partially upholds the policy.
#   3) "missing" — the assistant dives straight into
#      concrete portfolio advice on every turn without any
#      disclosure at all. The clear breach.
#
# `scenario` and `expected_outcome` are also read by the
# judge when present; populate them to give the judge
# more context about what the conversation was trying
# to do.

conversation_present = ConversationalTestCase(
    scenario=(
        "A user asks a series of investment questions; "
        "every reply must include both a 'not financial "
        "advice' statement and a recommendation to "
        "consult a licensed advisor."
    ),
    expected_outcome=(
        "Across every turn the assistant includes both "
        "required disclosures alongside its guidance."
    ),
    turns=[
        Turn(role="user", content="Should I move my savings into index funds?"),
        Turn(
            role="assistant",
            content=(
                "Index funds are a common low-cost option. "
                "This isn't financial advice, so please "
                "talk to a licensed advisor about your "
                "situation before you decide."
            ),
        ),
        Turn(role="user", content="How much should I put in international funds?"),
        Turn(
            role="assistant",
            content=(
                "Most investors keep international exposure "
                "between 20% and 40%. This isn't financial "
                "advice — please run the specifics past a "
                "licensed advisor."
            ),
        ),
    ],
    expected_labels={classifier.name: "present"},
)

conversation_partial = ConversationalTestCase(
    scenario=(
        "A user asks a series of investment questions; "
        "every reply must include both a 'not financial "
        "advice' statement and a recommendation to "
        "consult a licensed advisor."
    ),
    expected_outcome=(
        "The first reply is fully compliant; subsequent "
        "replies drop one or both disclosures, leaving the "
        "conversation only partially compliant overall."
    ),
    turns=[
        Turn(role="user", content="Should I move my savings into index funds?"),
        Turn(
            role="assistant",
            content=(
                "Index funds are a common low-cost option. "
                "This isn't financial advice, so please "
                "talk to a licensed advisor about your "
                "situation before you decide."
            ),
        ),
        Turn(role="user", content="How much should I put in international funds?"),
        Turn(
            role="assistant",
            content=(
                "Most investors keep international exposure "
                "between 20% and 40%. This isn't financial "
                "advice."
            ),
        ),
        Turn(role="user", content="And bonds?"),
        Turn(
            role="assistant",
            content=(
                "A 60/40 stocks-to-bonds split is a "
                "reasonable starting point — adjust for "
                "your risk tolerance and timeline."
            ),
        ),
    ],
    expected_labels={classifier.name: "partial"},
)

conversation_missing = ConversationalTestCase(
    scenario=(
        "A user asks a series of investment questions; "
        "every reply must include both a 'not financial "
        "advice' statement and a recommendation to "
        "consult a licensed advisor."
    ),
    expected_outcome=(
        "Across every turn the assistant gives concrete "
        "portfolio advice with no disclosure at all."
    ),
    turns=[
        Turn(role="user", content="Should I move my savings into index funds?"),
        Turn(
            role="assistant",
            content=(
                "Yes — put 70% into a total-market index "
                "fund and 30% into an international index "
                "fund, and rebalance once a year."
            ),
        ),
        Turn(role="user", content="And bonds?"),
        Turn(
            role="assistant",
            content=(
                "Hold 30% of your portfolio in bonds once "
                "you're within five years of retirement, "
                "and tilt toward short-duration funds as "
                "you get closer."
            ),
        ),
    ],
    expected_labels={classifier.name: "missing"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# All three conversations use the same disclosure
# checklist — only the assistant's behaviour changes.
# The judge applies the same rubric to all three, so
# the conversation-level disclosure rate reflects whether
# the assistant upheld the policy across the whole
# exchange, not a per-turn average.

evaluate(
    test_cases=[
        conversation_present,
        conversation_partial,
        conversation_missing,
    ],
    classifiers=[classifier],
)