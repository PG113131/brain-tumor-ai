from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate

# Directory containing prompt text files
PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    """
    Load a prompt from a text file.
    """
    file_path = PROMPTS_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {file_path}")

    return file_path.read_text(encoding="utf-8").strip()


# Load prompts
SYSTEM_PROMPT = load_prompt("system_prompt.txt")
USER_PROMPT_TEMPLATE = load_prompt("user_prompt.txt")
OUTPUT_PROMPT = load_prompt("output_prompt.txt")


def get_radiology_report_prompt() -> ChatPromptTemplate:
    """
    Returns the complete ChatPromptTemplate.
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT_TEMPLATE),
            ("user", OUTPUT_PROMPT),
        ]
    )