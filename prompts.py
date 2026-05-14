from pathlib import Path
import yaml
from langchain_core.prompts import ChatPromptTemplate

PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str) -> ChatPromptTemplate:
    path = PROMPTS_DIR / f"{name}.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    return ChatPromptTemplate.from_template(data["template"])