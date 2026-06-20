import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(
        self,
        embeddings: Embeddings,
        persist_path: str,
        collection_name: str,
    ) -> None:
        self.embeddings = embeddings
        self.persist_path = persist_path
        self.collection_name = collection_name

    def embed_and_store(self, chunks: List[Document]) -> Chroma:
        if not chunks:
            raise ValueError("No chunks provided for embedding.")

        logger.info(
            "Embedding %d chunk(s) into collection '%s' at %s",
            len(chunks),
            self.collection_name,
            self.persist_path,
        )

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_path,
        )

        logger.info("Indexed %d chunks successfully.", len(chunks))
        return vectorstore

    def load_store(self) -> Chroma:
        logger.info(
            "Loading existing vectorstore from '%s'", self.persist_path
        )
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_path,
        )
