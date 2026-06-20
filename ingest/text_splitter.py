import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def split(self, documents: List[Document]) -> List[Document]:
        if not documents:
            logger.warning("No documents provided for splitting.")
            return []

        chunks = self.splitter.split_documents(documents)
        logger.info(
            "Split %d document(s) into %d chunk(s).",
            len(documents),
            len(chunks),
        )
        return chunks
