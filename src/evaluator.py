import json

from pathlib import Path
from typing import List, TypeAlias

from .constants import RECALL_THRESHOLD
from .models import (
    AnsweredQuestion,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion
)

QList: TypeAlias = List[UnansweredQuestion | AnsweredQuestion]


class Evaluator:
    def __init__(
        self,
        student_path: str,
        correct_path: str
    ) -> None:
        self.student_path = student_path
        self.correct_path = correct_path

    def load_dataset(
        self,
        dataset_path: str
    ) -> RagDataset:
        dataset_file = Path(dataset_path)
        dataset_json = json.loads(dataset_file.read_text())
        return RagDataset(**dataset_json)

    def load_student(
        self,
        answers_path: str
    ) -> StudentSearchResults:
        dataset_file = Path(answers_path)
        dataset_json = json.loads(dataset_file.read_text())
        return StudentSearchResults(**dataset_json)

    def validate_context_length(
        self,
        dataset: List[MinimalSearchResults],
        length: int
    ) -> bool:
        for entry in dataset:
            for source in entry.retrieved_sources:
                if not (source.last_character_index
                        - source.first_character_index <= length):
                    return False
        return True

    def validate(
        self,
        max_context_length: int
    ) -> None:
        student = self.load_student(self.student_path).search_results
        correct = self.load_dataset(self.correct_path).rag_questions

        valid_len = self.validate_context_length(student, max_context_length)
        c1 = len({q.question_id for q in correct})
        c2 = len({q.question_id for q in correct if q.sources})
        c3 = len({q.question_id for q in student if q.retrieved_sources})

        print("Student data is valid:", valid_len)
        print("Total number of questions:", c1)
        print("Total number of questions with sources:", c2)
        print("Total number of questions with student sources:", c3)

    def count_evaluated(
        self
    ) -> int:
        student = self.load_student(self.student_path).search_results
        correct = self.load_dataset(self.correct_path).rag_questions
        count = 0

        for entry in student:
            truth = next((q for q in correct
                          if q.question_id == entry.question_id), None)
            if truth and getattr(truth, "sources", None):
                count += 1
        return count

    def is_source_found(
        self,
        source: MinimalSource,
        truth_source: MinimalSource
    ) -> bool:
        if (
            getattr(source, 'file_path', None)
            != getattr(truth_source, 'file_path', None)
        ):
            return False

        start_inter = max(source.first_character_index,
                          truth_source.first_character_index)
        end_inter = min(source.last_character_index,
                        truth_source.last_character_index)

        intersection = max(0, end_inter - start_inter)
        if intersection == 0:
            return False

        len_truth = (truth_source.last_character_index
                     - truth_source.first_character_index)
        if len_truth == 0:
            return False

        ratio = intersection / len_truth
        return ratio > RECALL_THRESHOLD

    def recallat(
        self,
        k: int
    ) -> float:
        student = self.load_student(self.student_path).search_results
        correct = self.load_dataset(self.correct_path).rag_questions

        if not correct:
            return 1.0

        scores = []
        truth_checked = 0

        for entry in student:
            truth = next((q for q in correct
                          if q.question_id == entry.question_id), None)

            if not truth or not truth.sources:
                continue

            truth_checked += 1

            if not entry.retrieved_sources:
                scores.append(0.0)
                continue

            unique_truth_found = set()
            for t_idx, truth_src in enumerate(truth.sources):
                for student_src in entry.retrieved_sources[:k]:
                    if self.is_source_found(student_src, truth_src):
                        unique_truth_found.add(t_idx)
                        break

            scores.append(len(unique_truth_found) / len(truth.sources))

        return sum(scores) / truth_checked if truth_checked > 0 else 0.0

    def evaluate(
        self,
        k: int
    ) -> None:
        count = self.count_evaluated()
        r1 = self.recallat(1)
        r3 = self.recallat(3)
        r5 = self.recallat(5)
        r10 = self.recallat(10)

        print("Evaluation Results")
        print("========================================")
        print("Questions evaluated:", count)
        print(f"Recall@1:  {r1:.3f}")
        print(f"Recall@3:  {r3:.3f}")
        print(f"Recall@5:  {r5:.3f}")
        print(f"Recall@10: {r10:.3f}")
