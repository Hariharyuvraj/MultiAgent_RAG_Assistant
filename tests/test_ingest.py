import shutil
import tempfile
from pathlib import Path

import pytest

from ingest.doc_loader import DocumentLoader
from ingest.text_splitter import DocumentSplitter


@pytest.fixture()
def sample_txt(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text("A" * 2000, encoding="utf-8")
    return tmp_path


def test_txt_loader_loads_documents(sample_txt):
    loader = DocumentLoader(str(sample_txt), ["txt"])
    docs = loader.load()
    assert len(docs) > 0


def test_text_splitter_chunks_correctly(sample_txt):
    loader = DocumentLoader(str(sample_txt), ["txt"])
    docs = loader.load()
    splitter = DocumentSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split(docs)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.page_content) <= 900


def test_loader_empty_directory(tmp_path):
    loader = DocumentLoader(str(tmp_path), ["pdf", "txt"])
    docs = loader.load()
    assert docs == []


def test_loader_missing_directory():
    loader = DocumentLoader("./nonexistent_dir", ["pdf"])
    with pytest.raises(FileNotFoundError):
        loader.load()
