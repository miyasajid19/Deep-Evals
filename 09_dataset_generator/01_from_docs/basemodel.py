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
        if schema is not None:
            # Augment prompt with JSON instructions so the model returns
            # a valid JSON object matching the schema — prompt-engineering
            # approach to JSON confinement for models without native
            # structured output support.
            schema_str = json.dumps(schema.model_json_schema(), indent=2)
            json_prompt = (
                f"{prompt}\n\n"
                f"You MUST respond with a valid JSON object that matches this schema:\n"
                f"{schema_str}\n"
                f"Respond with JSON only. No explanation, no markdown fences."
            )
            response = self.load_model().invoke(json_prompt)
        else:
            response = self.load_model().invoke(prompt)

        content = _strip_fences(response.content)

        if schema is not None:
            try:
                json_data = json.loads(content)
                return schema(**json_data)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Evaluation LLM outputted an invalid JSON. "
                    f"Raw output was: {content!r}"
                )

        return content

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str | BaseModel:
        if schema is not None:
            schema_str = json.dumps(schema.model_json_schema(), indent=2)
            json_prompt = (
                f"{prompt}\n\n"
                f"You MUST respond with a valid JSON object that matches this schema:\n"
                f"{schema_str}\n"
                f"Respond with JSON only. No explanation, no markdown fences."
            )
            response = await self.model.ainvoke(json_prompt)
        else:
            response = await self.model.ainvoke(prompt)

        content = _strip_fences(response.content)

        if schema is not None:
            try:
                json_data = json.loads(content)
                return schema(**json_data)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Evaluation LLM outputted an invalid JSON. "
                    f"Raw output was: {content!r}"
                )

        return content

    def get_model_name(self) -> str:
        return self.name

    def supports_multimodal(self) -> bool:
        # Set to True only if your MiniMax model actually supports images.
        return False


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name=os.getenv("MINIMAX_MODEL"),
        base_url=os.getenv("MINIMAX_BASE_URL"),
        api_key=os.getenv("MINIMAX_API_KEY"),
        extra_body={"thinking": {"type": "disabled"}},
    )


class OllamaEmbedding(OllamaEmbeddingModel):
    """Embedder backed by a local Ollama model.

    Defaults to `embeddinggemma:latest` (Google's EmbeddingGemma running
    through Ollama) so the synthesizer's context-construction pipeline
    (chunk embedding + cosine-similarity grouping) stays off OpenAI.
    Override `model` / `base_url` if your Ollama daemon lives elsewhere.
    """

    def __init__(
        self,
        model: str = "embeddinggemma:latest",
        base_url: Optional[str] = None,
    ):
        super().__init__(model=model, base_url=base_url)