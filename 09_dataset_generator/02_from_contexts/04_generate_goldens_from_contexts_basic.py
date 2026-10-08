from basemodel import CustomOpenAI
from deepeval.synthesizer import Synthesizer

# ==========================================
# 1. WHY FROM-CONTEXTS?
# ==========================================
# `generate_goldens_from_contexts` skips the document
# parsing / selection / grouping pipeline entirely.
# Hand it pre-built context groups (e.g. chunks you
# already pulled from your own vector DB) and it jumps
# straight to golden generation.
#
# Internally, `generate_goldens_from_docs` calls this
# method under the hood once context construction is
# done — so the golden format and grounding behavior
# are identical to from_docs, only the input shape
# differs.
#
# `contexts` shape: List[List[str]]. The OUTER list is
# the context groups, the INNER list is the chunks that
# make up each group. A group should share a theme so
# the synthesizer can ask coherent questions of it.

contexts = [
    [
        "All customers are eligible for a 30-day full refund at no extra cost.",
        "To request a refund, open the support page, enter the order number, and select 'Request refund'.",
        "Refunds are processed within five business days and appear on the original payment method.",
    ],
    [
        "Use the prepaid label in your order confirmation email to avoid extra charges.",
        "Customers are responsible for return shipping unless the item is damaged or incorrect.",
        "Drop the package at any authorized carrier location and keep the receipt until the refund is confirmed.",
    ],
]

# ==========================================
# 2. OPTIONAL: SOURCE FILES
# ==========================================
# If you want to track provenance on each generated
# golden, pass `source_files` with the same length as
# `contexts`. Each element can be a single string or a
# list of strings (when one context mixes multiple
# source files).

source_files = [
    ["../sample_doc.txt", "../sample_doc.txt", "../sample_doc.txt"],
    ["../sample_doc.txt", "../sample_doc.txt", "../sample_doc.txt"],
]

# ==========================================
# 3. THE CALL
# ==========================================
# One mandatory parameter (`contexts`) and three
# optional ones:
#   - include_expected_output (default True)
#   - max_goldens_per_context (default 2)
#   - source_files            (length must match `contexts`)

synthesizer = Synthesizer(model=CustomOpenAI())

goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=2,
    source_files=source_files,
)

# ==========================================
# 4. INSPECT
# ==========================================
# Each `Golden` carries `input`, `expected_output`, and
# `context` (the source group). `source_file` is set
# from `source_files` when provided.

for i, golden in enumerate(goldens):
    print(f"--- golden #{i} ---")
    print(f"input:           {golden.input}")
    print(f"expected_output: {golden.expected_output}")
    print(f"context[0..1]:   {golden.context[:2]}")
    print(f"source_file:     {golden.source_file}")