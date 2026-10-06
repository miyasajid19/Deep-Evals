import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from typing import Optional
from deepeval.models.base_model import DeepEvalBaseLLM

# Load environment variables
load_dotenv()

# ==========================================
# 1. DEFINE STRUCTURED CUSTOM LLM
# ==========================================
class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOpenAI(
            model_name=os.getenv("MINIMAX_MODEL"),
            base_url=os.getenv("MINIMAX_BASE_URL"),
            api_key=os.getenv("MINIMAX_API_KEY"),
    extra_body={"thinking": {"type": "disabled"}},  # Disables reasoning
            
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str | BaseModel:
        chat_model = self.load_model()
        if schema:
            structured_model = chat_model.with_structured_output(schema)
            return structured_model.invoke(prompt)
        
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None) -> str | BaseModel:
        chat_model = self.load_model()
        if schema:
            structured_model = chat_model.with_structured_output(schema)
            return await structured_model.ainvoke(prompt)
            
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Structured MiniMax Model"
    
def get_llm()->ChatOpenAI:
    return ChatOpenAI(
        model_name=os.getenv("MINIMAX_MODEL"),
        base_url=os.getenv("MINIMAX_BASE_URL"),
        api_key=os.getenv("MINIMAX_API_KEY"),
    extra_body={"thinking": {"type": "disabled"}},  # Disables reasoning
        
    )