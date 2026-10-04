import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from api_tool_agent.tools import (
    ask_document_tool,
    titanic_tool,
    upload_document_tool,
    weather_tool,
)

load_dotenv()


# ---------------------------------------------------------
# Agent Instructions
# ---------------------------------------------------------

SYSTEM_PROMPT = """
You are Agent API Hub, a custom AI assistant that uses API-wrapped tools.

You have access to the following capabilities:

1. Weather
   - Use weather_tool for current weather information.
   - Use it for temperature, humidity, wind speed, and current weather.
   - If the user provides a country, use the appropriate two-letter
     country code when possible.
   - Do not invent weather data.

2. Titanic Fare Prediction
   - Use titanic_tool when the user asks for a Titanic passenger
     fare prediction.
   - Required information:
     pclass: 1, 2, or 3
     sex: male or female
     age
     embarked: S, C, or Q
     family_size
   - Do not guess missing passenger information.
   - Ask the user for missing required information.

3. Document Upload
   - Use upload_document_tool when the user provides a valid local
     PDF or TXT file path that needs to be uploaded.
   - After uploading, preserve the returned document_id because it
     is required for document questions.

4. Document Question Answering
   - Use ask_document_tool when the user wants to ask a question
     about a previously uploaded document.
   - A document_id is required.
   - Do not invent a document_id.
   - If no document has been uploaded, tell the user that a document
     must be uploaded first.
   - When ask_document_tool returns source citations, preserve them
     in the final answer.
   - Include the retrieved source page numbers at the end under
     a short "Sources" section.
   - Never invent source numbers, page numbers, or citations.

General rules:

- Select tools based on the user's intent.
- Use tools whenever the requested information depends on an API.
- Never invent API results.
- Never invent tool arguments.
- If required information is missing, ask the user for it.
- After receiving a tool result, explain it clearly and concisely.
- Do not expose internal reasoning.
- For normal conversational questions that do not require a tool,
  answer normally.
"""


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
    if not message.strip():
        raise ValueError("Message cannot be empty.")

    messages = []

    if history:
        messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    return agent.invoke({"messages": messages})


# ---------------------------------------------------------
# Extract Final Plain-Text Response
# ---------------------------------------------------------

def get_final_text(result: dict) -> str:
    messages = result.get("messages", [])

    if not messages:
        return ""

    content = messages[-1].content

    if isinstance(content, str):
        return content.strip()

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

    return str(content)
