import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()


API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)


def _handle_response(response: httpx.Response) -> dict[str, Any]:
    """
    Validate an API response and return its JSON body.
    """

    try:
        response.raise_for_status()

    except httpx.HTTPStatusError as exc:
        try:
            error_data = response.json()
            detail = error_data.get("detail", str(exc))

        except Exception:
            detail = response.text or str(exc)

        raise RuntimeError(detail) from exc

    try:
        return response.json()

    except ValueError as exc:
        raise RuntimeError("API returned an invalid JSON response.") from exc


def upload_document(
    filename: str,
    file_bytes: bytes,
) -> dict[str, Any]:
    """
    Call POST /upload.

    Sends a PDF or TXT file to the API.
    """

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{API_BASE_URL}/upload",
                files={
                    "file": (
                        filename,
                        file_bytes,
                    )
                },
            )

        return _handle_response(response)

    except httpx.RequestError as exc:
        raise RuntimeError(f"Could not connect to API server: {exc}") from exc


def ask_document(
    document_id: str,
    question: str,
) -> dict[str, Any]:
    """
    Call POST /ask.

    Ask a question about an uploaded document.
    """

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{API_BASE_URL}/ask",
                json={
                    "document_id": document_id,
                    "question": question,
                },
            )

        return _handle_response(response)

    except httpx.RequestError as exc:
        raise RuntimeError(f"Could not connect to API server: {exc}") from exc


def get_weather(
    city: str,
    country_code: str | None = None,
) -> dict[str, Any]:
    """
    Call GET /weather.

    Retrieve current weather for a city.
    """

    params = {
        "city": city,
    }

    if country_code:
        params["country_code"] = country_code

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(
                f"{API_BASE_URL}/weather",
                params=params,
            )

        return _handle_response(response)

    except httpx.RequestError as exc:
        raise RuntimeError(f"Could not connect to API server: {exc}") from exc


def predict_titanic_fare(
    pclass: int,
    sex: str,
    age: float,
    embarked: str,
    family_size: int,
) -> dict[str, Any]:
    """
    Call POST /titanic.

    Predict a Titanic passenger's fare using
    the saved machine-learning model.
    """

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                f"{API_BASE_URL}/titanic",
                json={
                    "pclass": pclass,
                    "sex": sex,
                    "age": age,
                    "embarked": embarked,
                    "family_size": family_size,
                },
            )

        return _handle_response(response)

    except httpx.RequestError as exc:
        raise RuntimeError(f"Could not connect to API server: {exc}") from exc
