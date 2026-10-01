# Agent API Hub

> **A Custom AI Agent Using API-Wrapped Tools**

Agent API Hub is a custom AI agent built using **Python, Gemini, LangChain, FastAPI, httpx, Streamlit, and Scikit-learn**.

The project demonstrates how different capabilities can be exposed through API endpoints and then wrapped as tools that an AI agent can automatically select and execute based on a user's natural-language request.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Objective](#project-objective)
- [System Architecture](#system-architecture)
- [Features](#features)
- [API Endpoints](#api-endpoints)
- [LangChain Tools](#langchain-tools)
- [Automatic Tool Selection](#automatic-tool-selection)
- [Multi-Tool Execution](#multi-tool-execution)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Running the Project](#running-the-project)
- [Example Prompts](#example-prompts)
- [Example API Requests](#example-api-requests)
- [Error Handling](#error-handling)
- [Security](#security)
- [Current Limitations](#current-limitations)
- [Future Improvements](#future-improvements)
- [Key Learning Outcomes](#key-learning-outcomes)
- [Main Concept](#main-concept)

---

## Project Overview

The system provides four main API-based capabilities:

1. **Document Upload**
2. **Document Question Answering**
3. **Current Weather Lookup**
4. **Titanic Passenger Fare Prediction**

The user does not manually choose a tool.

Instead, the AI agent analyzes the user's request, understands the intent, selects the appropriate tool, calls the required API endpoint, receives the result, and produces a natural-language response.

---

## Project Objective

The main objective of this project is to demonstrate how an AI agent can use **API endpoints as tools**.

The general workflow is:

```text
User Request
    ↓
Gemini
    ↓
Intent Understanding
    ↓
LangChain Agent
    ↓
Automatic Tool Selection
    ↓
LangChain Tool
    ↓
Python API Wrapper
    ↓
httpx
    ↓
FastAPI Endpoint
    ↓
External Service / ML Model
    ↓
Tool Result
    ↓
Gemini
    ↓
Final Natural-Language Response
```

Example:

```text
User:
What is the current weather in Kathmandu, Nepal?

        ↓

Gemini understands that the request is about weather

        ↓

Agent selects weather_tool

        ↓

weather_tool calls the API wrapper

        ↓

API wrapper calls GET /weather

        ↓

FastAPI retrieves weather data from Open-Meteo

        ↓

Weather JSON is returned to the agent

        ↓

Gemini produces a readable response
```

---

## System Architecture

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │    Streamlit    │
                  │    Frontend     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ LangChain Agent │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     Gemini      │
                  │ 3.5 Flash Lite  │
                  └────────┬────────┘
                           │
                    Tool Selection
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
   Document Tools     Weather Tool     Titanic Tool
          │                │                │
          ▼                ▼                ▼
     API Wrapper       API Wrapper       API Wrapper
          │                │                │
          ▼                ▼                ▼
        httpx            httpx            httpx
          │                │                │
          ▼                ▼                ▼
     POST /upload      GET /weather     POST /titanic
     POST /ask
          │                │                │
          ▼                ▼                ▼
   Gemini Document     Open-Meteo       Scikit-learn
        Q&A               API              Model
```

---

# Features

## 1. Document Upload

The application supports uploading:

- PDF files
- TXT files

The backend reads the uploaded file, extracts readable text, stores the document temporarily in memory, and generates a unique `document_id`.

### Endpoint

```http
POST /upload
```

### Example Response

```json
{
  "message": "Document uploaded successfully.",
  "document_id": "47ed3e61-4b78-4cd2-b909-52c64878df0e",
  "filename": "example.pdf",
  "characters": 12540
}
```

The generated `document_id` is later used by the `/ask` endpoint.

---

## 2. Document Question Answering

After uploading a document, users can ask questions about it.

The system:

1. Receives the `document_id`
2. Retrieves the stored document
3. Gets the extracted document text
4. Sends the document context and question to Gemini
5. Returns a document-grounded answer

Gemini is instructed to answer only from the provided document and not invent information.

### Endpoint

```http
POST /ask
```

### Example Request

```json
{
  "document_id": "47ed3e61-4b78-4cd2-b909-52c64878df0e",
  "question": "Summarize this document in simple points."
}
```

### Example Prompt

```text
Summarize this document in simple points.
```

---

## 3. Weather Tool

The weather tool retrieves current weather information for a requested city.

The system first uses the **Open-Meteo Geocoding API** to find the latitude and longitude of the location.

It then uses those coordinates to request current weather information.

### Endpoint

```http
GET /weather
```

### Available Weather Information

The endpoint can return:

- City
- Country
- Country code
- Region
- Latitude
- Longitude
- Temperature
- Apparent temperature
- Humidity
- Wind speed
- Weather code
- Local time
- Timezone

### Example Request

```text
/weather?city=Kathmandu&country_code=NP
```

### Example Response

```json
{
  "city": "Kathmandu",
  "country": "Nepal",
  "country_code": "NP",
  "region": "Bagmati Province",
  "latitude": 27.70169,
  "longitude": 85.3206,
  "temperature_c": 22.1,
  "apparent_temperature_c": 25.3,
  "humidity_percent": 85,
  "wind_speed_kmh": 3.2,
  "weather_code": 51,
  "time": "2026-10-01T16:15",
  "timezone": "Asia/Kathmandu"
}
```

### Example User Prompt

```text
What is the current weather in Kathmandu, Nepal?
```

---

## 4. Titanic Fare Prediction

The Titanic tool uses a saved **Scikit-learn machine-learning pipeline** to estimate the fare of a Titanic passenger.

The trained model is loaded from:

```text
models/titanic_fare_regression_model.joblib
```

### Endpoint

```http
POST /titanic
```

### Model Inputs

The endpoint accepts:

- `pclass`
- `sex`
- `age`
- `embarked`
- `family_size`

The model internally expects:

```text
Pclass
Sex
Age
Embarked
FamilySize
IsAlone
```

`IsAlone` is calculated automatically from `FamilySize`.

If:

```text
FamilySize = 1
```

then:

```text
IsAlone = 1
```

Otherwise:

```text
IsAlone = 0
```

### Example Request

```json
{
  "pclass": 1,
  "sex": "female",
  "age": 25,
  "embarked": "S",
  "family_size": 1
}
```

### Example Response

```json
{
  "prediction_type": "Titanic passenger fare",
  "predicted_fare": 82.65,
  "passenger": {
    "pclass": 1,
    "sex": "female",
    "age": 25,
    "embarked": "S",
    "family_size": 1,
    "is_alone": 1
  }
}
```

### Example User Prompt

```text
Predict the Titanic fare for a 25-year-old female passenger in first class,
embarked from Southampton and travelling alone.
```

---

# API Endpoints

All required endpoints are implemented in a single FastAPI file.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/upload` | Upload PDF or TXT document |
| POST | `/ask` | Ask questions about an uploaded document |
| GET | `/weather` | Retrieve current weather |
| POST | `/titanic` | Predict Titanic passenger fare |

The API server is located at:

```text
src/api_tool_agent/api_server.py
```

---

# LangChain Tools

Each API endpoint is wrapped as a LangChain tool.

| LangChain Tool | API Endpoint |
|---|---|
| `upload_document_tool` | `POST /upload` |
| `ask_document_tool` | `POST /ask` |
| `weather_tool` | `GET /weather` |
| `titanic_tool` | `POST /titanic` |

The tools do not directly execute the backend logic.

Instead, the architecture is:

```text
LangChain Tool
      ↓
API Client Function
      ↓
httpx
      ↓
FastAPI Endpoint
      ↓
Actual Capability
```

This keeps the agent layer separate from the API implementation.

---

# Automatic Tool Selection

One of the main features of the project is automatic tool selection.

The user does not need to select:

```text
Weather
Titanic
Document
```

manually.

Gemini analyzes the user request and chooses the correct tool.

### Weather Example

```text
User:
What is the current weather in Kathmandu, Nepal?
```

The agent determines:

```text
Intent = Weather

Tool = weather_tool

Arguments:
city = Kathmandu
country_code = NP
```

Then:

```text
weather_tool
    ↓
GET /weather
    ↓
Weather Result
```

---

### Titanic Example

```text
User:
Predict the Titanic fare for a 25-year-old female passenger
in first class, embarked from Southampton and travelling alone.
```

The agent determines:

```text
Tool = titanic_tool

pclass = 1
sex = female
age = 25
embarked = S
family_size = 1
```

Then:

```text
titanic_tool
    ↓
POST /titanic
    ↓
Scikit-learn Model
    ↓
Predicted Fare
```

---

# Multi-Tool Execution

The agent can also perform more than one tool call for a single user request.

Example:

```text
Upload test_document.txt and tell me what Agent API Hub supports.
```

The agent can execute:

```text
User Request
    ↓
upload_document_tool
    ↓
POST /upload
    ↓
document_id
    ↓
ask_document_tool
    ↓
POST /ask
    ↓
Final Answer
```

This demonstrates that the agent can chain multiple API-wrapped tools together.

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| uv | Project and dependency management |
| Streamlit | Frontend interface |
| LangChain | Agent and tool framework |
| Gemini 3.5 Flash Lite | Large Language Model |
| FastAPI | Backend API server |
| httpx | HTTP API client |
| Pydantic | Request validation |
| PyPDF | PDF text extraction |
| Pandas | Titanic model input preparation |
| Scikit-learn | Machine-learning model |
| Joblib | Loading the saved ML model |
| Open-Meteo | Weather and geocoding data |
| python-dotenv | Environment variable management |
| Watchdog | Faster Streamlit file-change detection |

---

# Project Structure

```text
api-tool-agent/
│
├── app.py
├── README.md
├── .env
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
│
├── models/
│   └── titanic_fare_regression_model.joblib
│
└── src/
    └── api_tool_agent/
        │
        ├── __init__.py
        ├── agent.py
        ├── api_server.py
        │
        ├── api/
        │   ├── __init__.py
        │   └── client.py
        │
        ├── prompts/
        │   ├── __init__.py
        │   └── system_prompt.py
        │
        └── tools/
            ├── __init__.py
            ├── document_tool.py
            ├── weather_tool.py
            └── titanic_tool.py
```

---

# Installation

## 1. Clone the Repository

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd api-tool-agent
```

---

## 2. Install Dependencies

This project uses `uv`.

Run:

```bash
uv sync
```

This command installs all dependencies defined in:

```text
pyproject.toml
```

and:

```text
uv.lock
```

---

# Environment Configuration

Create your `.env` file from `.env.example`.

```bash
cp .env.example .env
```

The `.env.example` file contains:

```env
GEMINI_API_KEY=your_gemini_api_key_here
API_BASE_URL=http://127.0.0.1:8000
```

Open `.env` and add your real Gemini API key:

```env
GEMINI_API_KEY=your_actual_gemini_api_key
API_BASE_URL=http://127.0.0.1:8000
```

Do not commit the real `.env` file.

---

# Running the Project

The application requires two processes:

1. FastAPI Backend
2. Streamlit Frontend

---

## Terminal 1: Start FastAPI

Run:

```bash
uv run uvicorn api_tool_agent.api_server:app --reload --app-dir src
```

The API server will run at:

```text
http://127.0.0.1:8000
```

Swagger API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

Swagger should display:

```text
POST /upload
POST /ask
GET  /weather
POST /titanic
```

---

## Terminal 2: Start Streamlit

Open another terminal in the project directory.

Run:

```bash
uv run streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

---

# Example Prompts

## Weather

```text
What is the current weather in Kathmandu, Nepal?
```

```text
What is the temperature and humidity in Melbourne, Australia?
```

```text
What is the weather in New York, USA?
```

---

## Titanic

```text
Predict the Titanic fare for a 25-year-old female passenger in first class,
embarked from Southampton and travelling alone.
```

```text
Estimate the fare for a 30-year-old male passenger in third class,
embarked from Queenstown with a family size of 3.
```

---

## Document Questions

First upload a PDF or TXT file using the Streamlit sidebar.

Then ask:

```text
Summarize this document in simple points.
```

```text
What are the main topics discussed in this document?
```

```text
Explain the conclusion of this document.
```

```text
What does this document say about Retrieval-Augmented Generation?
```

---

# Example API Requests

## Upload Document

```bash
curl -X POST \
  "http://127.0.0.1:8000/upload" \
  -F "file=@example.pdf"
```

---

## Ask Document Question

```bash
curl -X POST \
  "http://127.0.0.1:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "YOUR_DOCUMENT_ID",
    "question": "Summarize this document."
  }'
```

---

## Get Weather

```bash
curl \
  "http://127.0.0.1:8000/weather?city=Kathmandu&country_code=NP"
```

---

## Predict Titanic Fare

```bash
curl -X POST \
  "http://127.0.0.1:8000/titanic" \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 25,
    "embarked": "S",
    "family_size": 1
  }'
```

---

# Example Agent Flows

## Weather Flow

```text
User
 ↓
"What is the weather in Kathmandu?"
 ↓
Gemini
 ↓
LangChain Agent
 ↓
weather_tool
 ↓
API Client
 ↓
GET /weather
 ↓
Open-Meteo
 ↓
Weather JSON
 ↓
Gemini
 ↓
Natural-Language Response
```

---

## Titanic Flow

```text
User
 ↓
Passenger Description
 ↓
Gemini
 ↓
LangChain Agent
 ↓
titanic_tool
 ↓
API Client
 ↓
POST /titanic
 ↓
Scikit-learn Pipeline
 ↓
Predicted Fare
 ↓
Gemini
 ↓
Natural-Language Response
```

---

## Document Flow

```text
User Uploads PDF
 ↓
POST /upload
 ↓
PDF Text Extraction
 ↓
document_id Generated
 ↓
User Asks Question
 ↓
Gemini Agent
 ↓
ask_document_tool
 ↓
POST /ask
 ↓
Document Context + Question
 ↓
Gemini
 ↓
Document-Grounded Response
```

---

# Error Handling

The project includes handling for common errors such as:

- Unsupported document types
- Empty uploaded documents
- Invalid TXT encoding
- Missing document IDs
- Empty questions
- Invalid weather locations
- Invalid country codes
- Weather API failures
- Missing Titanic model
- Invalid Titanic passenger inputs
- API connection failures
- Invalid JSON API responses
- Missing Gemini API key

---

# Security

The Gemini API key is stored inside:

```text
.env
```

The `.env` file is excluded from Git using `.gitignore`.

Example:

```gitignore
.env
```

Only the safe configuration template is committed:

```text
.env.example
```

Never commit:

- Gemini API keys
- Passwords
- Access tokens
- Private credentials

---

# .gitignore

Recommended `.gitignore` configuration:

```gitignore
# Python-generated files
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info/

# Virtual environment
.venv/

# Environment variables
.env

# macOS
.DS_Store
```

---

# Current Limitations

This project is designed as an assignment and learning project.

Current limitations include:

- Uploaded documents are stored only in memory.
- Restarting FastAPI removes previously generated `document_id` values.
- Document Q&A sends extracted document text directly to Gemini.
- No vector database is currently used.
- No persistent user authentication is implemented.
- Weather location resolution depends on Open-Meteo geocoding.
- The Titanic model predicts passenger fare, not survival.
- The system is currently designed for local development.
- The document system currently supports only PDF and TXT files.

---

# Future Improvements

Possible future improvements include:

- Persistent document storage
- Database integration
- Vector database support
- Full Retrieval-Augmented Generation pipeline
- Document chunking
- Embeddings
- Semantic search
- Multi-document question answering
- Persistent conversation memory
- User authentication
- Streaming responses
- Tool execution visualization
- API logging
- Automated tests
- Docker support
- Cloud deployment
- CI/CD
- Additional machine-learning tools
- Additional third-party API tools
- Multi-modal document support

---

# Key Learning Outcomes

This project demonstrates practical experience with:

- FastAPI API development
- REST API design
- API request validation
- Python `httpx`
- API wrapper functions
- LangChain tools
- Gemini tool calling
- Automatic agent routing
- Multi-tool execution
- AI agent design
- Document processing
- PDF text extraction
- External API integration
- Machine-learning model loading
- Scikit-learn pipelines
- Streamlit frontend development
- Conversation history
- Environment variable management
- Error handling
- Separation of concerns

---

# Main Concept

The central concept of the project is:

```text
API
 ↓
Python API Wrapper
 ↓
LangChain Tool
 ↓
AI Agent
 ↓
Automatic Tool Selection
```

### API

Provides the actual capability.

### Python API Wrapper

Uses `httpx` to communicate with the API.

### LangChain Tool

Exposes the API wrapper as a callable tool for the agent.

### AI Agent

Understands the user's natural-language request and determines which tool should be executed.

### Gemini

Handles language understanding, tool selection, and final response generation.

---

# Project Title

## Agent API Hub

### A Custom AI Agent Using API-Wrapped Tools