import os
import re
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from typing import Optional
from deepeval.models.base_model import DeepEvalBaseLLM

load_dotenv()

def _strip_fences(text: str) -> str:
    """Strip markdown ```json ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()

class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(self):
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
            # Manually parse JSON and validate against the Pydantic schema
            json_data = json.loads(content)
            return schema(**json_data)

        return content

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str | BaseModel:
        response = await self.model.ainvoke(prompt)
        content = _strip_fences(response.content)

        if schema is not None:
            # Manually parse JSON and validate against the Pydantic schema
            json_data = json.loads(content)
            return schema(**json_data)

        return content

    def get_model_name(self):
        return "Custom Structured MiniMax Model"


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model_name=os.getenv("MINIMAX_MODEL"),
        base_url=os.getenv("MINIMAX_BASE_URL"),
        api_key=os.getenv("MINIMAX_API_KEY"),
        extra_body={"thinking": {"type": "disabled"}},
    )