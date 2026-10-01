import os
import unicodedata
from io import BytesIO
from pathlib import Path
from typing import Literal
from uuid import uuid4

import httpx
import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()


app = FastAPI(
    title="Agent API Hub",
    description="API endpoints used as tools by the custom AI agent.",
    version="0.1.0",
)


# ---------------------------------------------------------
# Gemini
# ---------------------------------------------------------

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------------------------------------------------
# Titanic Model
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TITANIC_MODEL_PATH = PROJECT_ROOT / "models" / "titanic_fare_regression_model.joblib"

TITANIC_MODEL = None
TITANIC_MODEL_LOAD_ERROR = None

try:
    TITANIC_MODEL = joblib.load(TITANIC_MODEL_PATH)

except Exception as exc:
    TITANIC_MODEL_LOAD_ERROR = str(exc)


# ---------------------------------------------------------
# Temporary Document Storage
# ---------------------------------------------------------

DOCUMENT_STORE: dict[str, dict[str, str]] = {}


# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------


class AskRequest(BaseModel):
    document_id: str
    question: str


class TitanicFareRequest(BaseModel):
    pclass: Literal[1, 2, 3]

    sex: Literal["male", "female"]

    age: float = Field(
        gt=0,
        le=100,
        description="Passenger age",
    )

    embarked: Literal["S", "C", "Q"]

    family_size: int = Field(
        ge=1,
        le=20,
        description="Total family members including the passenger",
    )


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------


def normalize_location_name(value: str) -> str:
    """
    Normalize location names.

    Example:
    Tansen and Tānsen will be treated as the same name.
    """

    normalized = unicodedata.normalize("NFKD", value)

    return (
        "".join(char for char in normalized if not unicodedata.combining(char))
        .strip()
        .casefold()
    )


# ---------------------------------------------------------
# 1. POST /upload
# ---------------------------------------------------------


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT document and extract its text.
    """

    filename = file.filename or ""
    extension = Path(filename).suffix.lower()

    if extension not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    try:
        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if extension == ".pdf":
            reader = PdfReader(BytesIO(file_bytes))

            extracted_pages = []

            for page in reader.pages:
                page_text = page.extract_text()

                if page_text:
                    extracted_pages.append(page_text)

            text = "\n".join(extracted_pages)

        else:
            text = file_bytes.decode("utf-8")

        text = text.strip()

        if not text:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the document.",
            )

        document_id = str(uuid4())

        DOCUMENT_STORE[document_id] = {
            "filename": filename,
            "text": text,
        }

        return {
            "message": "Document uploaded successfully.",
            "document_id": document_id,
            "filename": filename,
            "characters": len(text),
        }

    except HTTPException:
        raise

    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="TXT file must use UTF-8 encoding.",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to process document: {exc}",
        )


# ---------------------------------------------------------
# 2. POST /ask
# ---------------------------------------------------------


@app.post("/ask")
async def ask_document(request: AskRequest):
    """
    Ask a question about a previously uploaded document.
    """

    document = DOCUMENT_STORE.get(request.document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found. Upload the document first.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    document_text = document["text"]

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using only the information contained
in the provided document.

If the answer cannot be found in the document, say:

"I could not find that information in the document."

Do not invent information.

DOCUMENT:
{document_text}

QUESTION:
{question}
"""

    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
        )

        answer = response.text

        if not answer:
            raise HTTPException(
                status_code=500,
                detail="Gemini returned an empty response.",
            )

        return {
            "document_id": request.document_id,
            "filename": document["filename"],
            "question": question,
            "answer": answer.strip(),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to answer question: {exc}",
        )


# ---------------------------------------------------------
# 3. GET /weather
# ---------------------------------------------------------


@app.get("/weather")
async def get_weather(
    city: str,
    country_code: str | None = None,
):
    """
    Get current weather information for a city.

    Examples:
    NP = Nepal
    US = United States
    AU = Australia
    """

    city = city.strip()

    if not city:
        raise HTTPException(
            status_code=400,
            detail="City name cannot be empty.",
        )

    if country_code:
        country_code = country_code.strip().upper()

        if len(country_code) != 2:
            raise HTTPException(
                status_code=400,
                detail="country_code must be a 2-letter ISO country code.",
            )

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:

            geocoding_params = {
                "name": city,
                "count": 10,
                "language": "en",
                "format": "json",
            }

            if country_code:
                geocoding_params["countryCode"] = country_code

            geocoding_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params=geocoding_params,
            )

            geocoding_response.raise_for_status()

            location_data = geocoding_response.json()

            results = location_data.get("results")

            if not results:
                location_description = city

                if country_code:
                    location_description += f" ({country_code})"

                raise HTTPException(
                    status_code=404,
                    detail=f"Location '{location_description}' was not found.",
                )

            normalized_city = normalize_location_name(city)

            exact_match = next(
                (
                    result
                    for result in results
                    if normalize_location_name(result.get("name", ""))
                    == normalized_city
                ),
                None,
            )

            if exact_match is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Exact location '{city}' was not found"
                        + (f" in {country_code}." if country_code else ".")
                        + " Try using a nearby city or municipality name."
                    ),
                )

            location = exact_match

            latitude = location["latitude"]
            longitude = location["longitude"]

            weather_response = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": (
                        "temperature_2m,"
                        "relative_humidity_2m,"
                        "apparent_temperature,"
                        "weather_code,"
                        "wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
            )

            weather_response.raise_for_status()

            weather_data = weather_response.json()

            current = weather_data.get("current")

            if not current:
                raise HTTPException(
                    status_code=502,
                    detail="Weather service returned no current weather data.",
                )

            return {
                "city": location.get("name"),
                "country": location.get("country"),
                "country_code": location.get("country_code"),
                "region": location.get("admin1"),
                "latitude": latitude,
                "longitude": longitude,
                "temperature_c": current.get("temperature_2m"),
                "apparent_temperature_c": current.get("apparent_temperature"),
                "humidity_percent": current.get("relative_humidity_2m"),
                "wind_speed_kmh": current.get("wind_speed_10m"),
                "weather_code": current.get("weather_code"),
                "time": current.get("time"),
                "timezone": weather_data.get("timezone"),
            }

    except HTTPException:
        raise

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Weather service request failed: {exc}",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to retrieve weather: {exc}",
        )


# ---------------------------------------------------------
# 4. POST /titanic
# ---------------------------------------------------------


@app.post("/titanic")
async def predict_titanic_fare(request: TitanicFareRequest):
    """
    Predict Titanic passenger fare using the saved ML model.
    """

    if TITANIC_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Titanic model could not be loaded. " f"{TITANIC_MODEL_LOAD_ERROR}"
            ),
        )

    try:
        # IsAlone is derived from FamilySize.
        is_alone = 1 if request.family_size == 1 else 0

        # Column names MUST match the names used during model training.
        passenger_data = pd.DataFrame(
            [
                {
                    "Pclass": request.pclass,
                    "Sex": request.sex,
                    "Age": request.age,
                    "Embarked": request.embarked,
                    "FamilySize": request.family_size,
                    "IsAlone": is_alone,
                }
            ]
        )

        predicted_fare = TITANIC_MODEL.predict(passenger_data)[0]

        return {
            "prediction_type": "Titanic passenger fare",
            "predicted_fare": round(
                float(predicted_fare),
                2,
            ),
            "passenger": {
                "pclass": request.pclass,
                "sex": request.sex,
                "age": request.age,
                "embarked": request.embarked,
                "family_size": request.family_size,
                "is_alone": is_alone,
            },
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unable to predict Titanic fare: {exc}",
        )
