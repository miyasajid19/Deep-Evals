from typing import List
from pydantic import BaseModel, RootModel
from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import JsonCorrectnessMetric

# ==========================================
# 1. THE METRIC — list-of-objects schema
# ==========================================
# When `actual_output` is a JSON ARRAY of records,
# wrap your `BaseModel` in `RootModel[List[Model]]`
# so pydantic validates every element against the
# same shape. The metric accepts either a single
# object or a list — what matters is that the
# supplied schema matches the JSON's top-level
# structure.
#
# Note: `RootModel` is the pydantic v2 spelling.
# On pydantic v1 use `__root__: List[Foo]`.

class Item(BaseModel):
    sku: str
    price: float

class ItemList(RootModel[Item]):
    pass

# ==========================================
# 2. THE METRIC INSTANCE
# ==========================================
# The metric passes the entire JSON through pydantic;
# the validator walks the list and asserts every item
# has `sku: str` and `price: float`.

list_metric = JsonCorrectnessMetric(
    expected_schema=ItemList,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 3. VALID LIST — score 1
# ==========================================

valid_list_case = LLMTestCase(
    input="List 2 inventory items as JSON",
    actual_output=(
        '[{"sku": "A-001", "price": 9.99}, '
        '{"sku": "B-002", "price": 14.50}]'
    ),
)
list_metric.measure(valid_list_case)
print("--- valid list ---")
print(f"Score:   {list_metric.score}")           # 1.0
print(f"Reason:  {list_metric.reason}")
print(f"Passed:  {list_metric.is_successful()}")  # True

# ==========================================
# 4. ONE BAD ITEM IN THE LIST — score 0
# ==========================================
# First item is fine; second item has `price`
# spelled as a string. pydantic catches it during
# list traversal, the whole output fails.

mixed_case = LLMTestCase(
    input="List 2 inventory items as JSON",
    actual_output=(
        '[{"sku": "A-001", "price": 9.99}, '
        '{"sku": "B-002", "price": "fourteen-fifty"}]'
    ),
)
list_metric.measure(mixed_case)
print("--- one bad item ---")
print(f"Score:   {list_metric.score}")            # 0.0
print(f"Reason:  {list_metric.reason}")
print(f"Passed:  {list_metric.is_successful()}")  # False

# ==========================================
# 5. WHEN TO REACH FOR LIST SCHEMAS
# ==========================================
# Use a `RootModel[List[Foo]]` whenever the LLM
# emits "give me N examples", "list all matches",
# "show top-3" — anything where the natural shape
# is a list. Pairing the schema with a `Goldens`
# dataset keeps every list length and item shape
# testable in CI.