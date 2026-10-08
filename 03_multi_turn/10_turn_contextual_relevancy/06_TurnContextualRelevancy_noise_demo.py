from basemodel import CustomOpenAI
from deepeval.test_case import Turn, ConversationalTestCase
from deepeval.metrics import TurnContextualRelevancyMetric

# ==========================================
# 1. THE METRIC
# ==========================================
# A single-metric run on three retrieval setups that
# share the same user question but differ in how much of
# the retrieval_context is on-topic. Each noise statement
# drags the score down — even when the assistant's reply
# is correct, because the metric grades the retriever.

turn_contextual_relevancy = TurnContextualRelevancyMetric(
    threshold=0.5,
    model=CustomOpenAI(),
    include_reason=True,
)

# ==========================================
# 2. SHARED QUESTION + REPLY
# ==========================================
user_turn = Turn(role="user", content="What if these shoes don't fit?")

assistant_content = "We offer a 30-day full refund at no extra cost."

# ==========================================
# 3. CASE A — clean retrieval
# ==========================================
# All statements relevant. Expect score 1.0.

clean_case = ConversationalTestCase(
    turns=[
        user_turn,
        Turn(
            role="assistant",
            content=assistant_content,
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost.",
            ],
        ),
    ]
)

# ==========================================
# 4. CASE B — half the retrieval is noise
# ==========================================
# Two statements: one relevant, one noise. Expect 0.5.

half_noise_case = ConversationalTestCase(
    turns=[
        user_turn,
        Turn(
            role="assistant",
            content=assistant_content,
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost. Our company was founded in 1998.",
            ],
        ),
    ]
)

# ==========================================
# 5. CASE C — mostly noise
# ==========================================
# Three statements: one relevant, two noise. Expect 1/3 ≈ 0.333.

mostly_noise_case = ConversationalTestCase(
    turns=[
        user_turn,
        Turn(
            role="assistant",
            content=assistant_content,
            retrieval_context=[
                "All customers are eligible for a 30 day full refund at no extra cost. "
                "Our company was founded in 1998. "
                "We also sell socks, hats, and jackets."
            ],
        ),
    ]
)

# ==========================================
# 6. RUN THE METRIC ON ALL THREE CASES
# ==========================================
# Same user question, same correct answer — only the
# retrieval_context quality changes.

def run(label, case):
    turn_contextual_relevancy.measure(case)
    print(f"--- {label} ---")
    print(f"  Score:  {turn_contextual_relevancy.score:.3f}")
    print(f"  Passed: {turn_contextual_relevancy.is_successful()}")
    print(f"  Reason: {turn_contextual_relevancy.reason}")

run("A — clean retrieval",     clean_case)
run("B — half noise",          half_noise_case)
run("C — mostly noise",        mostly_noise_case)