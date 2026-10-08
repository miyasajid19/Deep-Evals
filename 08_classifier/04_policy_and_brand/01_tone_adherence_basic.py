from basemodel import CustomOpenAI
from deepeval.classifiers import ToneAdherenceClassifier
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# ==========================================
# 1. THE CLASSIFIER
# ==========================================
# `ToneAdherenceClassifier` is a built-in, categorical
# LLM-as-a-judge shipped by DeepEval for one specific
# question: does the assistant's reply match the voice it
# was configured to have? Its `name` and label set are
# fixed — that's the point of a built-in classifier, the
# labels mean the same thing across every dataset and
# run, so tone-adherence rates are comparable between
# evaluations.
#
# The `tone` argument describes the voice: a string the
# judge folds into the label descriptions so the same
# classifier measures against any voice you configure.
# Without `tone` the judge relies on the test case alone,
# which is fine for generic "warm vs cold" judgments but
# blunt for branded voices.
#
# The two labels cover both directions:
#
#   - "on_tone"  — the response matches the configured tone
#                  in wording, register, and length.
#   - "off_tone" — the response departs from the configured
#                  tone in wording, register, or length.
#
# Tone is product: a brand voice is part of the user
# experience, and a prompt or model change that shifts
# how the assistant sounds is a regression that shows up
# as "off_tone" rate rather than slipping through
# unnoticed.
#
# Optional constructor knobs (defaults shown):
#   - tone                   — a string describing the
#                              configured voice, folded
#                              into the label descriptions.
#                              Defaulted to None; without
#                              it the judge relies on the
#                              test case alone.
#   - model                  — evaluation LLM (defaults to
#                              DeepEval's default GPT
#                              model). Pass a
#                              `DeepEvalBaseLLM` to use
#                              your own model, e.g. the
#                              `CustomOpenAI` defined in
#                              `basemodel.py` for this repo.
#   - include_reason=True    — when True, the judge also
#                              returns a short reason
#                              explaining the chosen label.
#   - allow_none=False       — when False (the default), the
#                              judge always picks the closest
#                              label. Set True to let the
#                              judge abstain (`label=None`
#                              with a reason) when none fit.
#   - async_mode=True        — runs `classify()` concurrently
#                              across test cases.
#   - classification_template — override the judge prompt.
#   - eval_mode              — "llm" (default) or "system_one"
#                              (Jev-as-a-judge, requires
#                              `TYPESAFE_API_KEY`).

classifier = ToneAdherenceClassifier(
    tone="warm, plain-spoken, and under three sentences",
    model=CustomOpenAI(),
)

# ==========================================
# 2. THE TEST CASES
# ==========================================
# `expected_labels` is a dict keyed by `classifier.name`.
# For `ToneAdherenceClassifier`, that key is fixed — the
# built-in `name` DeepEval assigns to this classifier —
# so you set the expected label as a string the judge
# will return.
#
# We exercise both label values against the same configured
# voice, with four replies that pair the same user
# complaint against different assistant styles:
#
#   - "on_tone" (two cases) — a short, warm, plain-spoken
#     acknowledgement that keeps the response under three
#     sentences. The brand voice. Both are positive
#     examples because tone regressions are usually
#     directional — a model can drift toward formal,
#     toward verbose, or toward chirpy all at once — and
#     you want to test for the shape, not the length.
#   - "off_tone" (two cases) — the same user complaint
#     answered in styles that miss the configured voice.
#     One goes legalistic and paragraph-long, the other
#     goes upbeat but slips in product marketing copy.
#     Both fail for different reasons; both should fail.
#
# Pairing the same prompt with different replies is the
# key: a single prompt is not a tone test, the
# *comparison* is.
#
# `evaluate()` runs the judge on each case, prints a
# report, and writes it to the configured sink (Confident
# AI if connected, else the local cache). The case PASSES
# for this classifier when the returned label matches the
# expected one.

test_case_on_tone_short = LLMTestCase(
    input="My package hasn't arrived.",
    actual_output=(
        "Sorry about that! I've checked and it's out for "
        "delivery today. I'll keep an eye on it for you."
    ),
    expected_labels={classifier.name: "on_tone"},
)

test_case_on_tone_brief = LLMTestCase(
    input="My package hasn't arrived.",
    actual_output=(
        "Apologies for the wait — it's out for delivery "
        "right now. I'll flag it for follow-up if it slips."
    ),
    expected_labels={classifier.name: "on_tone"},
)

test_case_off_tone_formal = LLMTestCase(
    input="My package hasn't arrived.",
    actual_output=(
        "Dear valued customer,\n\nThank you for your "
        "recent inquiry regarding the status of your "
        "outbound shipment. Pursuant to our standard "
        "delivery terms (Section 4.2 of the Customer "
        "Service Agreement), please be advised that "
        "consignments may experience transit delays "
        "due to factors outside the carrier's direct "
        "control. We respectfully request your patience "
        "while our logistics partners complete the "
        "scheduled delivery cycle. Should the package "
        "fail to arrive within five (5) business days "
        "from the date of this correspondence, kindly "
        "re-engage via the support channel and reference "
        "ticket #49281. We remain at your service for "
        "any further clarification you may require."
    ),
    expected_labels={classifier.name: "off_tone"},
)

test_case_off_tone_marketing = LLMTestCase(
    input="My package hasn't arrived.",
    actual_output=(
        "OMG NO 😱 I'm SO sorry!! Our delivery heroes are "
        "literally sprinting to your door as we speak — "
        "have you tried our brand-new Premium Express "
        "add-on? It's a game-changer and our customers "
        "literally cannot live without it!! 💖💖💖"
    ),
    expected_labels={classifier.name: "off_tone"},
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# All four cases use the same user complaint and the same
# configured voice — only the assistant's reply changes.
# The judge applies the same rubric to all four, so the
# tone-adherence rate is comparable across model versions
# and prompt revisions without rebuilding the rubric each
# run.

evaluate(
    test_cases=[
        test_case_on_tone_short,
        test_case_on_tone_brief,
        test_case_off_tone_formal,
        test_case_off_tone_marketing,
    ],
    classifiers=[classifier],
)