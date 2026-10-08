from basemodel import CustomOpenAI
from deepeval.test_case import LLMTestCase
from deepeval.metrics import HallucinationMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single metric instance scoring three responses
# against multiple contexts at varying levels of
# consistency. The score is the share of contexts that
# stayed consistent — higher is better.

hallucination = HallucinationMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. CASE A — all three contexts aligned
# ==========================================
# Context [1] location, [2] clothing, [3] activity —
# all consistent. Expect 3/3 = 1.0.

all_aligned = LLMTestCase(
    input="Tell me about the man.",
    actual_output="A blond man in a brown shirt drinking water from a public fountain on the south side of the park.",
    context=[
        "The fountain is on the south side of the park, near the bench.",
        "He was wearing a brown shirt.",
        "He was drinking water from a public fountain.",
    ],
)

# ==========================================
# 3. CASE B — two of three aligned (color contradiction)
# ==========================================
# Location and activity aligned, but the shirt color
# contradicts context [2]. Expect 2/3 ≈ 0.667.

color_contradicted = LLMTestCase(
    input="Tell me about the man.",
    actual_output="A blond man in a red shirt drinking water from a public fountain on the south side of the park.",
    context=[
        "The fountain is on the south side of the park, near the bench.",
        "He was wearing a brown shirt.",
        "He was drinking water from a public fountain.",
    ],
)

# ==========================================
# 4. CASE C — only location aligned (full contradiction)
# ==========================================
# Both clothing and activity wrong. Only location
# aligned. Expect 1/3 ≈ 0.333.

only_location = LLMTestCase(
    input="Tell me about the man.",
    actual_output="A blond man in a red shirt reading a book on the south side of the park.",
    context=[
        "The fountain is on the south side of the park, near the bench.",
        "He was wearing a brown shirt.",
        "He was drinking water from a public fountain.",
    ],
)

# ==========================================
# 5. RUN THE METRIC ON ALL THREE CASES
# ==========================================

def run(label, case):
    hallucination.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {hallucination.score:.3f}")
    print(f"  Passed: {hallucination.is_successful()}")
    print(f"  Reason: {hallucination.reason}")

run("A — all three aligned",         all_aligned)
run("B — two of three aligned",      color_contradicted)
run("C — only location aligned",     only_location)