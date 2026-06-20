import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

logger = logging.getLogger(__name__)


class DocumentLoader:
    SUPPORTED = {".pdf": PyPDFLoader, ".txt": TextLoader}

    def __init__(self, docs_path: str, supported_formats: List[str]) -> None:
        self.docs_path = Path(docs_path)
        self.extensions = {f".{fmt.lstrip('.')}" for fmt in supported_formats}

    def load(self) -> List[Document]:
        if not self.docs_path.exists():
            raise FileNotFoundError(f"Documents directory not found: {self.docs_path}")

        documents: List[Document] = []
        files = [
            f for f in self.docs_path.iterdir()
            if f.is_file() and f.suffix.lower() in self.extensions
        ]

        if not files:
            logger.warning("No supported files found in %s", self.docs_path)
            return documents

        for file_path in files:
            loader_cls = self.SUPPORTED.get(file_path.suffix.lower())
            if loader_cls is None:
                logger.warning("No loader for extension %s — skipping %s", file_path.suffix, file_path.name)
                continue
            try:
                loader = loader_cls(str(file_path))
                docs = loader.load()
                for doc in docs:
                    doc.metadata.setdefault("source", file_path.name)
                    doc.metadata.setdefault("format", file_path.suffix.lstrip("."))
                documents.extend(docs)
                logger.info("Loaded %d page(s) from %s", len(docs), file_path.name)
            except Exception:
                logger.exception("Failed to load %s", file_path.name)

        logger.info("Total documents loaded: %d", len(documents))
        return documents
