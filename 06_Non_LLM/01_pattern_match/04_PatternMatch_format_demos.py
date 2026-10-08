from deepeval.test_case import LLMTestCase
from deepeval.metrics import PatternMatchMetric

# ==========================================
# 1. THE METRIC — common shape patterns
# ==========================================
# PatternMatchMetric is the right tool whenever the
# output must follow a SHAPE rather than a fixed
# string — IDs, dates, phone numbers, hex codes.
# This demo walks four canonical shapes:
#
#   - ISO date             2025-12-31
#   - UUID v4              8-4-4-4-12 hex
#   - Hex color            #RRGGBB
#   - Major.minor.patch    1.2.3 (semver)

# ==========================================
# 2. ONE PATTERN-MATCH METRIC PER SHAPE
# ==========================================

iso_date = PatternMatchMetric(
    pattern=r"^\d{4}-\d{2}-\d{2}$",
    threshold=1.0,
)

uuid_v4 = PatternMatchMetric(
    # 8-4-4-4-12 hex; the 4th group starts with 8/9/a/b
    pattern=(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-"
        r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
    ),
    ignore_case=True,
    threshold=1.0,
)

hex_color = PatternMatchMetric(
    pattern=r"^#[0-9a-fA-F]{6}$",
    threshold=1.0,
)

semver = PatternMatchMetric(
    pattern=r"^\d+\.\d+\.\d+$",
    threshold=1.0,
)

# ==========================================
# 3. SAMPLE OUTPUTS — one good, one bad per shape
# ==========================================

shape_checks = [
    (iso_date,    "What date is Christmas 2025?",
                  "2025-12-25",    "12/25/2025"),
    (uuid_v4,     "Mint a UUID v4",
                  "550e8400-e29b-41d4-a716-446655440000",
                  "550e8400-e29b-41d4-c716-446655440000"),  # 'c' invalid
    (hex_color,   "Pick a random hex color",
                  "#1f9d55",       "#1f9d5"),
    (semver,      "Latest stable version?",
                  "2.7.1",         "2.7"),
]

# ==========================================
# 4. SCORE EACH
# ==========================================
# Each metric is run twice — once against a
# well-formed sample, once against a malformed one.

for metric, prompt, good, bad in shape_checks:
    metric.measure(LLMTestCase(input=prompt, actual_output=good))
    print(f"good={good!s:>40s}  score={metric.score}")
    metric.measure(LLMTestCase(input=prompt, actual_output=bad))
    print(f"bad ={bad!s:>40s}  score={metric.score}")
    print()