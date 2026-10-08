from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics.community import CitationFaithfulnessMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# CitationFaithfulnessMetric is a COMMUNITY metric — it
# lives in `deepeval.metrics.community`, not
# `deepeval.metrics`. It checks whether every `[N]`
# citation marker in `actual_output` points to the
# passage in `retrieval_context` that actually supports
# the claim the marker is attached to.
#
# It is STRICTER than FaithfulnessMetric: faithfulness
# only asks "is this claim supported by SOMETHING in
# the context?", citation faithfulness additionally
# catches the case where a claim cites [1] but [1] doesn't
# support it, even if some other passage would.
#
# Required: `input`, `actual_output`, `retrieval_context`.
# `retrieval_context` passages are numbered ([1], [2], ...)
# before the judge sees them, and the `[N]` markers in
# the output refer to those passage numbers.
#
# Threshold defaults to 1.0 (any unfaithful verdict
# fails). Score is binary pass/fail: 1.0 = faithful,
# 0.0 = unfaithful.

# ==========================================
# 2. THE TEST CASE — misattributed citation
# ==========================================
# The completion-year claim cites [1], which only
# covers height. The correct supporting passage is [2].
# A FaithfulnessMetric would pass this (the claim IS
# supported by something in the context); a
# CitationFaithfulnessMetric fails it.

citation_faithfulness = CitationFaithfulnessMetric(
    threshold=1.0,
    model=CustomOpenAI(),
    include_reason=True,
)

test_case = LLMTestCase(
    input="How tall is the Eiffel Tower and when was it completed?",
    actual_output="The Eiffel Tower was completed in 1889 [1].",
    retrieval_context=[
        "The Eiffel Tower stands 330 metres tall in Paris.",
        "The Eiffel Tower was completed in 1889 for the World Fair.",
    ],
)

# ==========================================
# 3. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and
# writes a report (Confident AI if configured, else
# local cache).

evaluate(test_cases=[test_case], metrics=[citation_faithfulness])