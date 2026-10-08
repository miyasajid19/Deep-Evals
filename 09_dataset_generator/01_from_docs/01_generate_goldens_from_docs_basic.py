from basemodel import CustomOpenAI, OllamaEmbedding
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

# ==========================================
# 1. THE SYNTHESIZER
# ==========================================
# `generate_goldens_from_docs` is the right entry point
# when you have raw documents (.txt, .docx, .pdf, .md,
# .markdown, .mdx) and want deepeval to do the context
# extraction for you. The synthesizer runs an internal
# context-construction pipeline before golden generation:
#
#   1) Document parsing — split each document into
#      token-based chunks via `TokenTextSplitter`
#      (`chunk_size`, `chunk_overlap`).
#   2) Context selection — pick random nodes and
#      quality-score them with the `critic_model`.
#   3) Context grouping — group selected nodes with
#      similar neighbors (cosine similarity) to form
#      coherent contexts.
#
# Each constructed context then drives the same golden
# generation pipeline used by `generate_goldens_from_contexts`.

synthesizer = Synthesizer(model=CustomOpenAI())

# ==========================================
# 2. THE CALL
# ==========================================
# One mandatory parameter (`document_paths`) and three
# optional ones:
#   - include_expected_output   (default True)
#   - max_goldens_per_context  (default 2)
#   - context_construction_config (default ContextConstructionConfig())
#
# Final golden count ~= max_goldens_per_context *
#                       max_contexts_per_document
# (not just max_goldens_per_context).

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=["D:/DeepEvals/09_dataset_generator/01_from_docs/sample_doc.txt",r"F:\Sajid_Miya_AI_Engineer_resume.pdf"],
    include_expected_output=True,
    max_goldens_per_context=1,
    # Override the default OpenAI embedder so chunk
    # embedding + similarity grouping both run through
    # local Ollama instead of `text-embedding-3-small`.
    context_construction_config=ContextConstructionConfig(
        embedder=OllamaEmbedding(),
        critic_model=CustomOpenAI(),
    ),
)

# ==========================================
# 3. INSPECT THE OUTPUT
# ==========================================
# Each `Golden` has `input`, `expected_output`, and
# `context`. `context` is the exact chunk-group the
# synthesizer built from the source document — keep it
# around so you can plumb it into your RAG's retrieval
# step when you evaluate the system end-to-end.

# Save as CSV
synthesizer.save_as(
    file_type="csv",
    directory="./synthetic_data",
    file_name="my_dataset",  # saves as my_dataset.csv
)

# Save as JSON
synthesizer.save_as(
    file_type="json",
    directory="./synthetic_data",
    file_name="my_dataset",  # saves as my_dataset.json
)

# Save as JSONL
synthesizer.save_as(
    file_type="jsonl",
    directory="./synthetic_data",
    file_name="my_dataset",  # saves as my_dataset.jsonl
)



# from deepeval.dataset import EvaluationDataset

# dataset = EvaluationDataset(goldens=synthesizer.synthetic_goldens)
# dataset.push(alias="My Generated Dataset")

for i, golden in enumerate(goldens[:5]):
    print(f"--- golden #{i} ---")
    from rich import print
    print(golden.dict())