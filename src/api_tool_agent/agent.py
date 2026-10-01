import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from api_tool_agent.prompts.system_prompt import SYSTEM_PROMPT
from api_tool_agent.tools.document_tool import (
    ask_document_tool,
    upload_document_tool,
)
from api_tool_agent.tools.titanic_tool import titanic_tool
from api_tool_agent.tools.weather_tool import weather_tool

load_dotenv()


# ---------------------------------------------------------
# Gemini API Key
# ---------------------------------------------------------

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to the .env file.")


# ---------------------------------------------------------
# Gemini Model
# ---------------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=api_key,
)


# ---------------------------------------------------------
# Available LangChain Tools
# ---------------------------------------------------------

TOOLS = [
    weather_tool,
    titanic_tool,
    upload_document_tool,
    ask_document_tool,
]


# ---------------------------------------------------------
# Create Agent
# ---------------------------------------------------------

agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


# ---------------------------------------------------------
# Run Agent
# ---------------------------------------------------------


def run_agent(
    message: str,
    history: list[dict] | None = None,
) -> dict:
    """
    Send a user message to the agent.

    Optional conversation history can be provided so
    the agent can understand previous messages.

    The agent decides:
    - whether a tool is required
    - which tool to use
    - what arguments to send
    - whether multiple tools are required
    """

    if not message.strip():
        raise ValueError("Message cannot be empty.")

    messages = []

    # Add previous conversation messages
    if history:
        messages.extend(history)

    # Add the current user message
    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    # Execute the LangChain agent
    result = agent.invoke({"messages": messages})

    return result


# ---------------------------------------------------------
# Extract Final Plain-Text Response
# ---------------------------------------------------------


def get_final_text(result: dict) -> str:
    """
    Extract the final assistant response as plain text.

    Gemini/LangChain may return:
    - a normal string
    - structured content blocks

    This function converts both formats into plain text.
    """

    messages = result.get("messages", [])

    if not messages:
        return ""

    final_message = messages[-1]

    content = final_message.content

    # Case 1: Normal string response
    if isinstance(content, str):
        return content.strip()

    # Case 2: Structured Gemini content blocks
    if isinstance(content, list):
        text_parts = []

        for block in content:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "text":
                text = block.get("text", "")

                if text:
                    text_parts.append(text)

        return "\n".join(text_parts).strip()

    # Fallback
    return str(content)
