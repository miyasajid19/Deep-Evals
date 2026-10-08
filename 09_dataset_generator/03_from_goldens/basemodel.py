import os
import re
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from typing import Optional
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.models import OllamaEmbeddingModel

load_dotenv()

def _strip_fences(text: str) -> str:
    """Strip markdown ```json ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()

class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(self):
        # Do NOT call super().__init__() — it calls load_model() before
        # self.model is assigned, which would raise an AttributeError.
        # Instead, set self.name manually to satisfy DeepEval's attribute check.
        self.name = "MiniMax-M3"
        self.model = ChatOpenAI(
            model_name=os.getenv("MINIMAX_MODEL"),
            base_url=os.getenv("MINIMAX_BASE_URL"),
            api_key=os.getenv("MINIMAX_API_KEY"),
            extra_body={"thinking": {"type": "disabled"}},
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str | BaseModel:
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        content = _strip_fences(response.content)

        if schema is not None:
            json_data = json.loads(content)
            return schema(**json_data)

        return content

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str | BaseModel:
        response = await self.model.ainvoke(prompt)
        content = _strip_fences(response.content)

        if schema is not None:
            json_data = json.loads(content)
            return schema(**json_data)

        return content

    def get_model_name(self) -> str:
        return self.name

    def supports_multimodal(self) -> bool:
        # Required for multimodal test cases that include MLLMImage objects.
        # Only set to True if your underlying MiniMax model actually supports images.
        return True


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name=os.getenv("MINIMAX_MODEL"),
        base_url=os.getenv("MINIMAX_BASE_URL"),
        api_key=os.getenv("MINIMAX_API_KEY"),
        extra_body={"thinking": {"type": "disabled"}},
    )


class OllamaEmbedding(OllamaEmbeddingModel):
    """Embedder backed by a local Ollama model.

    Defaults to `embeddinggemma:latest` (Google's
    EmbeddingGemma running through Ollama) so the
    synthesizer's context-construction pipeline
    (chunk embedding + cosine-similarity grouping) stays
    off OpenAI. Override `model` / `base_url` if your
    Ollama daemon lives elsewhere.
    """

    def __init__(
        self,
        model: str = "embeddinggemma:latest",
        base_url: Optional[str] = None,
    ):
        super().__init__(model=model, base_url=base_url)