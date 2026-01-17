"""Document loader with compatibility across llama-index versions.

It attempts to use `SimpleDirectoryReader` when available. If the
import fails (API changed), it falls back to a simple reader that
creates `Document` objects from TXT and PDF files.
"""

import os
import tempfile
from typing import List

try:
    from llama_index.core import SimpleDirectoryReader
except Exception:
    SimpleDirectoryReader = None  # type: ignore

try:
    from llama_index.core import Document
except Exception:
    Document = None  # type: ignore

import pypdf


class DocumentLoader:
    @staticmethod
    def _read_pdf(path: str) -> str:
        try:
            reader = pypdf.PdfReader(path)
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            return "\n".join(texts)
        except Exception:
            return ""

    @staticmethod
    def load_files(uploaded_files) -> List:
        """Save uploaded files to a temp directory and load as Documents.

        Uses `SimpleDirectoryReader` when available; otherwise reads
        `.txt` and `.pdf` files manually and wraps them in
        `llama_index.Document` objects when possible.
        """
        documents = []
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded files to temp directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # If SimpleDirectoryReader is available, prefer it
            if SimpleDirectoryReader:
                return SimpleDirectoryReader(temp_dir).load_data()

            # Fallback: walk files and create Document objects (or raw dicts)
            for root, _, files in os.walk(temp_dir):
                for fname in files:
                    path = os.path.join(root, fname)
                    content = ""
                    if fname.lower().endswith(".pdf"):
                        content = DocumentLoader._read_pdf(path)
                    else:
                        try:
                            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                                content = fh.read()
                        except Exception:
                            content = ""

                    if Document:
                        documents.append(Document(text=content, metadata={"file_name": fname}))
                    else:
                        documents.append({"text": content, "file_name": fname})

        return documents