from pathlib import Path

from langchain_core.tools import tool

from api_tool_agent.client import (
    ask_document,
    get_weather,
    predict_titanic_fare,
    upload_document,
)


@tool
def weather_tool(
    city: str,
    country_code: str | None = None,
) -> dict:
    """
    Get the current weather for a city.

    Use this tool when the user asks about current weather,
    temperature, humidity, wind speed, or weather conditions.

    Args:
        city:
            City or municipality name.

        country_code:
            Optional two-letter ISO country code.
            Examples: NP, US, AU.
    """

    return get_weather(
        city=city,
        country_code=country_code,
    )


@tool
def titanic_tool(
    pclass: int,
    sex: str,
    age: float,
    embarked: str,
    family_size: int,
) -> dict:
    """
    Predict the estimated Titanic passenger fare.

    Args:
        pclass:
            Passenger class: 1, 2, or 3.

        sex:
            "male" or "female".

        age:
            Passenger age in years.

        embarked:
            S = Southampton
            C = Cherbourg
            Q = Queenstown

        family_size:
            Total family size including the passenger.
    """

    return predict_titanic_fare(
        pclass=pclass,
        sex=sex,
        age=age,
        embarked=embarked,
        family_size=family_size,
    )


@tool
def upload_document_tool(file_path: str) -> dict:
    """
    Upload a local PDF or TXT document.

    Args:
        file_path:
            Path to the local PDF or TXT file.
    """

    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if path.suffix.lower() not in {".pdf", ".txt"}:
        raise ValueError("Only PDF and TXT files are supported.")

    return upload_document(
        filename=path.name,
        file_bytes=path.read_bytes(),
    )


@tool
def ask_document_tool(
    document_id: str,
    question: str,
) -> dict:
    """
    Ask a question about a previously uploaded document.

    Args:
        document_id:
            ID returned after the document was uploaded.

        question:
            Question about the uploaded document.
    """

    return ask_document(
        document_id=document_id,
        question=question,
    )
