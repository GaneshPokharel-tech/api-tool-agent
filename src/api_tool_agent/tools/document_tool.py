from pathlib import Path

from langchain_core.tools import tool

from api_tool_agent.api.client import (
    ask_document,
    upload_document,
)


@tool
def upload_document_tool(file_path: str) -> dict:
    """
    Upload a PDF or TXT document to the document API.

    Use this tool when a document needs to be uploaded before
    questions can be asked about it.

    Args:
        file_path:
            Path to a local PDF or TXT file.
    """

    path = Path(file_path)

    if not path.exists():
        raise ValueError(f"File does not exist: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if path.suffix.lower() not in {".pdf", ".txt"}:
        raise ValueError("Only PDF and TXT files are supported.")

    file_bytes = path.read_bytes()

    return upload_document(
        filename=path.name,
        file_bytes=file_bytes,
    )


@tool
def ask_document_tool(
    document_id: str,
    question: str,
) -> dict:
    """
    Ask a question about a previously uploaded document.

    Use this tool when the user wants information from an
    uploaded PDF or TXT document.

    Args:
        document_id:
            ID returned by the upload_document_tool.

        question:
            Question to ask about the uploaded document.
    """

    return ask_document(
        document_id=document_id,
        question=question,
    )
