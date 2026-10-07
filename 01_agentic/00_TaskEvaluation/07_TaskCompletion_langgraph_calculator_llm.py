import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import TaskCompletionMetric
from deepeval.integrations.langchain import CallbackHandler

load_dotenv()


# ==========================================
# 1. DEFINE CALCULATOR TOOLS
# ==========================================

@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


@tool
def divide(a: float, b: float) -> float:
    """Divide a by b. Raises if b is zero."""
    if b == 0:
        raise ValueError("division by zero")
    return a / b


TOOLS = [add, subtract, multiply, divide]


# ==========================================
# 2. BUILD THE LANGGRAPH AGENT
# ==========================================

llm = ChatOpenAI(
    model_name=os.getenv("MINIMAX_MODEL"),
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
    extra_body={"thinking": {"type": "disabled"}},
).bind_tools(TOOLS)


def chatbot(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


graph = (
    StateGraph(MessagesState)
    .add_node(chatbot)
    .add_node("tools", ToolNode(TOOLS))
    .add_edge(START, "chatbot")
    .add_conditional_edges("chatbot", tools_condition)
    .add_edge("tools", "chatbot")
    .compile()
)


# ==========================================
# 3. BUILD A DATASET
# ==========================================

dataset = EvaluationDataset(
    goldens=[
        Golden(input="What is 17 + 25?"),
        Golden(input="What is 144 divided by 12?"),
        Golden(input="What is (8 x 9) minus 30?"),
    ]
)


# ==========================================
# 4. DEFINE THE METRIC
# ==========================================
from basemodel import CustomOpenAI
task_completion = TaskCompletionMetric(
    threshold=0.7,
    include_reason=True,
    model=CustomOpenAI(),
)


# ==========================================
# 5. RUN THE EVALS ITERATOR
# ==========================================

for golden in dataset.evals_iterator(metrics=[task_completion]):
    graph.invoke(
        {"messages": [{"role": "user", "content": golden.input}]},
        config={"callbacks": [CallbackHandler()]},
    )