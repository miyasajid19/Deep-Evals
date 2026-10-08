from deepeval.test_case import LLMTestCase
from deepeval.metrics import PatternMatchMetric

# ==========================================
# 1. THE METRIC — fullmatch vs match semantics
# ==========================================
# PatternMatchMetric uses `re.fullmatch`, which
# anchors the pattern at BOTH ends of the input.
# That means `\d+` does NOT match "The code is 1234"
# — the digits are only a substring, not the whole
# string.
#
# Three rules to remember:
#   1. The pattern must consume every character
#      of `actual_output`.
#   2. Implicit anchors: no need to write ^ and $,
#      but adding them is harmless.
#   3. To match a fragment INSIDE the output, wrap
#      the relevant part with `.*` on each side:
#      `r".*\d+.*"` matches anything that contains
#      at least one digit.

# ==========================================
# 2. STRICT METRIC — only the digits
# ==========================================
# Will pass for "1234" but fail for anything with
# surrounding text.

digits_only = PatternMatchMetric(
    pattern=r"\d+",          # fullmatch is implicit
    threshold=1.0,
    verbose_mode=True,
)

# ==========================================
# 3. CASE A — bare digits — score 1
# ==========================================
bare = LLMTestCase(
    input="Give me the verification code as digits",
    actual_output="1234",
)
digits_only.measure(bare)
print("--- '1234' ---")
print(f"Score:   {digits_only.score}")           # 1.0
print(f"Reason:  {digits_only.reason}")

# ==========================================
# 4. CASE B — text wrapping the digits — score 0
# ==========================================
# Same digits, but the LLM added a prefix. Without
# `.*` in the pattern, fullmatch fails.
wrapped = LLMTestCase(
    input="Give me the verification code as digits",
    actual_output="The code is 1234",
)
digits_only.measure(wrapped)
print("--- 'The code is 1234' ---")
print(f"Score:   {digits_only.score}")            # 0.0
print(f"Reason:  {digits_only.reason}")

# ==========================================
# 5. LOOSEN THE PATTERN — wrap with .*
# ==========================================
# Adding `.*` on each end allows any prefix and
# suffix, so any string CONTAINING digits passes.

digits_inside = PatternMatchMetric(
    pattern=r".*\d+.*",     # any prefix + digits + any suffix
    threshold=1.0,
)

digits_inside.measure(wrapped)
print("--- 'The code is 1234' with .*\\d+.* ---")
print(f"Score:   {digits_inside.score}")           # 1.0
print(f"Reason:  {digits_inside.reason}")

# ==========================================
# 6. TWO COMMON PITFALLS
# ==========================================
# a) `pattern=r"\d+"` looks like it should "find
#    digits" — but fullmatch means the WHOLE output
#    must be digits. Use `r".*\d+.*"` to allow
#    surrounding text.
#
# b) `pattern=r"abc"` only matches the literal
#    three-character output "abc". For "abc123"
#    you'd need `r"abc.*"` (or `r"abc"` with
#    `re.search` semantics, which is NOT what this
#    metric uses).
#
# When in doubt, set `verbose_mode=True` and look
# at the printed pattern + reason. The reason
# echoes the regex back at you so you can compare it
# against the input side-by-side.