from pydantic import BaseModel
from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import JsonCorrectnessMetric
from deepeval import evaluate

# ==========================================
# 1. THE METRIC
# ==========================================
# JsonCorrectnessMetric validates whether
# `actual_output` (a JSON string) loads cleanly into
# a `pydantic` `BaseModel` schema you supply via
# `expected_schema`. Scoring is deterministic and
# free of token cost.
#
# Score formula:
#   1  if the JSON parses AND every required field
#      has the right type
#   0  otherwise
#
# The `model` argument is ONLY used to generate a
# human-readable reason when the output fails the
# schema AND `include_reason=True`. Set
# `include_reason=False` to skip the LLM call
# entirely — the metric itself does not need a model.

# ==========================================
# 2. THE EXPECTED SCHEMA
# ==========================================
# Define a `BaseModel` that captures the fields and
# constraints you want enforced. Types, required vs
# optional fields, and validators all flow through.

class ExampleSchema(BaseModel):
    name: str

# ==========================================
# 3. THE METRIC INSTANCE
# ==========================================
# `threshold=0.5` is the deepeval default; for a
# schema gate metric the score is binary (0 or 1), so
# 0.5 behaves identically to 1.0 in practice.

json_correct = JsonCorrectnessMetric(
    expected_schema=ExampleSchema,
    model=CustomOpenAI(),     # only used for the failure reason
    include_reason=True,
)

# ==========================================
# 4. THE TEST CASE — valid JSON
# ==========================================
# The LLM was asked for a JSON object with a 'name'
# field, and produced `"name": "Alice"` — a valid
# JSON string that loads into ExampleSchema.

test_case = LLMTestCase(
    input="Output me a random Json with the 'name' key",
    actual_output='{"name": "Alice"}',
)

# ==========================================
# 5. RUN THE EVAL
# ==========================================
# `evaluate()` runs the metric, prints results, and
# writes a report (Confident AI if configured, else
# local cache).

evaluate(test_cases=[test_case], metrics=[json_correct])