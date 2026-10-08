from basemodel import CustomOpenAI, OllamaEmbedding
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

# ==========================================
# 1. CUSTOM CONTEXT CONSTRUCTION
# ==========================================
# `ContextConstructionConfig` is the only knob you tune
# at the call site (not at Synthesizer construction time)
# because context construction is unique to the
# `from_docs` entry point. Knobs that matter:
#
#   chunk_size / chunk_overlap
#     Token-based chunking. Smaller chunks => more
#     candidates; very large chunks => may fail to produce
#     max_contexts_per_document unique nodes (the
#     synthesizer raises on that).
#
#   max_contexts_per_document / min_contexts_per_document
#     How many distinct context groups to build from
#     each document.
#
#   max_context_length / min_context_length
#     How many chunks each context contains.
#
#   context_quality_threshold
#     Minimum LLM-assigned quality score (clarity,
#     depth, structure, relevance) for a selected node
#     to survive filtering. Nodes that fail max_retries
#     times still get used (best-of) so generation
#     never starves.
#
#   context_similarity_threshold
#     Cosine-similarity floor for grouping two nodes
#     into the same context. Same retry-with-best-of
#     behavior.
#
#   max_retries
#     Retry budget for both selection and grouping.
#
#   allow_cross_file_contexts / max_files_per_context /
#   target_files_per_context
#     Multi-file grouping controls. With
#     `allow_cross_file_contexts=False` (default) every
#     context is built from a single file.
#
#   critic_model / embedder
#     Override the judge model for quality scoring and
#     the embedder for similarity grouping.

context_construction_config = ContextConstructionConfig(
    critic_model=CustomOpenAI(),
    embedder=OllamaEmbedding(),
    max_contexts_per_document=3,
    min_contexts_per_document=1,
    max_context_length=3,
    min_context_length=1,
    chunk_size=512,
    chunk_overlap=64,
    context_quality_threshold=0.5,
    context_similarity_threshold=0.5,
    max_retries=3,
    allow_cross_file_contexts=False,
)

# ==========================================
# 2. RUN
# ==========================================
# Same call as the basic example, just with the custom
# config threaded in.

synthesizer = Synthesizer(model=CustomOpenAI(),async_mode=False,max_concurrent=3)

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=["C:/Users/miyas/Downloads/Sorting_Algos.pdf","C:/Users/miyas/Downloads/UMS_report.docx","C:/Users/miyas/Downloads/scaler.txt"],
    include_expected_output=True,
    max_goldens_per_context=2,
    context_construction_config=context_construction_config,
)


# Save as CSV
synthesizer.save_as(
    file_type="csv",
    directory="./synthetic_data",
    file_name="my_dataset2",  # saves as my_dataset.csv
)

# Save as JSON
synthesizer.save_as(
    file_type="json",
    directory="./synthetic_data",
    file_name="my_dataset2",  # saves as my_dataset.json
)

# Save as JSONL
synthesizer.save_as(
    file_type="jsonl",
    directory="./synthetic_data",
    file_name="my_dataset2",  # saves as my_dataset.jsonl
)



print(f"Generated {len(goldens)} goldens")
for i, golden in enumerate(goldens[:3]):
    print(f"--- golden #{i} ---")
    print(f"input:           {golden.input}")
    print(f"expected_output: {golden.expected_output}")