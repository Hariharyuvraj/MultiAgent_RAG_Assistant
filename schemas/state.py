from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryInput(BaseModel):
    query: str
    session_id: str
    timestamp: str


class ChunkResult(BaseModel):
    content: str
    source: str
    page: int
    score: float


class WebResult(BaseModel):
    title: str
    body: str
    url: str


class AgentState(BaseModel):
    query: str
    session_id: str
    history: List[dict] = Field(default_factory=list)
    enriched_query: str = ""
    plan: List[str] = Field(default_factory=list)
    retrieved_chunks: List[ChunkResult] = Field(default_factory=list)
    answer: str = ""
    sources: List[str] = Field(default_factory=list)
    grounding_score: float = 0.0
    retry_count: int = 0
    passed_guard: bool = False
    latency_ms: Optional[int] = None
    is_web_answer: bool = False
    web_sources: List[WebResult] = Field(default_factory=list)
    is_blocked: bool = False
    block_category: str = ""

    def to_session_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_session_dict(cls, data: dict) -> "AgentState":
        return cls(**data)
