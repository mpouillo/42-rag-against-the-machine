from pydantic import BaseModel, Field
import uuid
from typing import List, Sequence


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

    def __hash__(self):
        return hash((self.file_path,
                     self.first_character_index,
                     self.last_character_index))


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: Sequence[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: Sequence[MinimalAnswer]
