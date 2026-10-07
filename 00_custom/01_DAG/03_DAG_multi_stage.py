from basemodel import CustomOpenAI
from deepeval.metrics.dag import (
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    TaskNode,
    DeepAcyclicGraph,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import DAGMetric
from deepeval import evaluate
from rich import print

# ==========================================
# 1. DEFINE A TASK NODE + TWO SHARED JUDGEMENTS
# ==========================================
# A TaskNode first extracts headings from the actual output. Both the
# "presence" and "order" judgements reuse that extracted evidence, so we
# connect the task to each judgement with add_node().

extract = TaskNode(
    instructions="Extract every heading from the actual output, in order.",
    output_label="Extracted headings",
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
    label="Heading extraction",
)

headings_present = BinaryJudgementNode(
    criteria="Are all required headings present in the extracted headings?",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT],
    label="Required headings",
)

heading_order = NonBinaryJudgementNode(
    criteria="Classify the ordering of the extracted headings.",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT],
    label="Heading order",
)

# Share the extracted output with both downstream judgements
extract.add_node(headings_present)
extract.add_node(heading_order)

# ==========================================
# 2. WIRE UP VERDICTS
# ==========================================
# If headings are missing → fail immediately.
# If headings are present → continue to the ordering judgement.
headings_present.add_verdict(verdict=False, score=0)
headings_present.add_verdict(verdict=True, then=heading_order)

heading_order.add_verdict(verdict="Correct order", score=10)
heading_order.add_verdict(verdict="Partially out of order", score=5)
heading_order.add_verdict(verdict="Incorrect order", score=0)

# ==========================================
# 3. BUILD DAG METRIC
# ==========================================
metric = DAGMetric(
    name="Document Structure",
    dag=DeepAcyclicGraph(root_nodes=[extract]),
    model=CustomOpenAI(),
    threshold=0.5,
    include_reason=True,
    verbose_mode=True,
)

# ==========================================
# 4. TEST CASES
# ==========================================
test_cases = [
    LLMTestCase(
        input="List headings for a quarterly report.",
        expected_output="Introduction, Revenue, Expenses, Conclusion",
        actual_output=(
            "# Introduction\n"
            "Quarterly overview.\n\n"
            "# Revenue\n"
            "Total revenue: $1.2M.\n\n"
            "# Expenses\n"
            "Operating costs.\n\n"
            "# Conclusion\n"
            "Strong quarter."
        ),
    ),
    LLMTestCase(
        input="List headings for a quarterly report.",
        expected_output="Introduction, Revenue, Expenses, Conclusion",
        actual_output=(
            "# Revenue\n"
            "Total revenue: $1.2M.\n\n"
            "# Introduction\n"
            "Quarterly overview.\n\n"
            "# Expenses\n"
            "Operating costs.\n\n"
            "# Conclusion\n"
            "Strong quarter."
        ),
    ),
    LLMTestCase(
        input="List headings for a quarterly report.",
        expected_output="Introduction, Revenue, Expenses, Conclusion",
        actual_output=(
            "# Introduction\n"
            "Quarterly overview.\n\n"
            "# Revenue\n"
            "Total revenue: $1.2M.\n\n"
            "# Summary\n"
            "Recap of the quarter."
        ),
    ),
]

# ==========================================
# 5. EVALUATE
# ==========================================
evaluate(test_cases=test_cases, metrics=[metric])