from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from server.rag.retriever import retrieve_examples
from server.prompts import load_prompt
from dotenv import load_dotenv
import json

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

# --- Topic routing sets ---

POLYNOMIAL_TOPICS = {"factoring", "roots", "expansion", "polynomial_operations"}
QUESTION_TOPICS = {"definitions", "graphing", "word_problems", "theory"}


# --- Player analysis ---

def analyze_player(data: dict) -> tuple[str, str]:
    accuracy = data["accuracy"]
    mistakes = data["recent_mistakes"]
    focus_topic = max(set(mistakes), key=mistakes.count)

    if accuracy < 0.5:
        difficulty = "easy"
    elif accuracy < 0.75:
        difficulty = "normal"
    elif accuracy < 0.9:
        difficulty = "hard"
    else:
        difficulty = "very_hard"

    return difficulty, focus_topic


# --- Tools ---

@tool
def retrieve_polynomial_examples(topic: str, difficulty: str, mistakes: list[str], k: int = 5) -> str:
    """
    Retrieves relevant polynomial examples from the knowledge base.
    Use this before generating a polynomial problem.

    Args:
        topic: The math topic to focus on (e.g. 'factoring', 'roots')
        difficulty: One of 'easy', 'normal', 'hard', 'very_hard'
        mistakes: List of recent mistake types the player made
        k: Number of examples to retrieve
    """
    examples = retrieve_examples(topic=topic, difficulty=difficulty, mistakes=mistakes, k=k)
    return "\n\n".join(examples) if examples else "No examples found."


def extract_text(content) -> str:
    if isinstance(content, list):
        return content[0]["text"]
    return content

@tool
def generate_polynomial(
    accuracy: float,
    avg_time: float,
    topic: str,
    difficulty: str,
    examples: str
) -> str:
    """
    Generates a polynomial problem for the player given their stats and retrieved examples.
    Always call retrieve_polynomial_examples first and pass the result here.

    Args:
        accuracy: Player's current accuracy (0.0 to 1.0)
        avg_time: Player's average time per question in seconds
        topic: The math topic to focus on
        difficulty: One of 'easy', 'normal', 'hard', 'very_hard'
        examples: Retrieved examples string from retrieve_polynomial_examples
    """
    polynomial_chain = load_prompt("polynomial") | llm
    response = polynomial_chain.invoke({
        "accuracy": accuracy,
        "avg_time": avg_time,
        "topic": topic,
        "difficulty": difficulty,
        "examples": examples
    })
    return extract_text(response.content)


@tool
def generate_question(accuracy: float, avg_time: float, topic: str, difficulty: str) -> str:
    """
    Generates a conceptual or multiple choice question for the player.
    Use this when mode is 'question'. Does not require retrieval.

    Args:
        accuracy: Player's current accuracy (0.0 to 1.0)
        avg_time: Player's average time per question in seconds
        topic: The math topic to focus on
        difficulty: One of 'easy', 'normal', 'hard', 'very_hard'
    """
    question_chain = load_prompt("question") | llm
    response = question_chain.invoke({
        "accuracy": accuracy,
        "avg_time": avg_time,
        "topic": topic,
        "difficulty": difficulty,
    })
    return extract_text(response.content)

# --- Agent setup ---

SYSTEM_PROMPT = """You are a math education AI that generates personalized problems for players.

You will be told explicitly whether to generate a POLYNOMIAL or a QUESTION via the 'mode' field in the input.

If mode is 'polynomial':
  1. Call retrieve_polynomial_examples with the given topic, difficulty, and mistakes
  2. Call generate_polynomial with the player stats and the retrieved examples

If mode is 'question':
  1. Call generate_question directly with the player stats
  2. Do NOT call retrieve_polynomial_examples

Always follow the mode exactly. Do not switch between modes."""

tools = [retrieve_polynomial_examples, generate_polynomial, generate_question]

agent = create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


# --- Entry point ---

def run_agent(player_data: dict) -> dict:
    difficulty, topic = analyze_player(player_data)

    mode = player_data.get("force_mode") or (
        "polynomial" if topic in POLYNOMIAL_TOPICS else "question"
    )

    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"""
        Player stats:
        - Accuracy: {player_data['accuracy']}
        - Avg time per question: {player_data['avg_time']}s
        - Recent mistakes: {player_data['recent_mistakes']}
        - Topic: {topic}
        - Difficulty: {difficulty}
        - Mode: {mode}

        Generate a {mode} for this player.
        """
        }]
    })

    # Walk messages in reverse to find the last tool result
    tool_output = None
    for message in reversed(result["messages"]):
        role = getattr(message, "type", None) or getattr(message, "role", None)
        if role == "tool":
            tool_output = extract_text(message.content)
            break

    # Fall back to the last message if no tool message found
    if tool_output is None:
        tool_output = extract_text(result["messages"][-1].content)

    try:
        parsed = json.loads(tool_output)
    except json.JSONDecodeError:
        parsed = tool_output
    
    print(parsed)

    return {
        "raw": parsed,
        "difficulty": difficulty,
        "topic": topic,
        "mode": mode
    }