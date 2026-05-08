from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7
)

def analyze_player(data):
    accuracy = data["accuracy"]
    mistakes = data["recent_mistakes"]

    # Find most frequent mistake
    focus_topic = max(set(mistakes), key=mistakes.count)

    if accuracy < 0.5:
        difficulty = "easy"
    elif accuracy < 0.75:
        difficulty = "medium"
    else:
        difficulty = "hard"

    return difficulty, focus_topic

prompt = ChatPromptTemplate.from_template("""
You are generating math exercises for a student.

Student profile:
- Accuracy: {accuracy}
- Average time: {avg_time} seconds
- Weak topic: {topic}

Generate THREE factorization exercises.

Requirements:
- Focus on the weak topic: {topic}
- Difficulty: {difficulty}
- Polynomial in x
- Factorable over integers

Topics:
- common_factor
- perfect_square_trinomial
- difference_of_squares
- sum_product (for quadratic polynomials)

Return ONLY the polynomials to be factorised, separated by a comma, without additional characters such as $.
""")

chain = prompt | llm

def generate_exercise(player_data):
    difficulty, topic = analyze_player(player_data)

    response = chain.invoke({
        "accuracy": player_data["accuracy"],
        "avg_time": player_data["avg_time"],
        "topic": topic,
        "difficulty": difficulty
    })

    return {
        "raw": response.content,
        "difficulty": difficulty,
        "topic": topic
    }