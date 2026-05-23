from typing import List, TypeAlias

from .constants import RECALL_THRESHOLD
from .IOUtils import IOUtils
from .models import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    UnansweredQuestion
)

QList: TypeAlias = List[UnansweredQuestion | AnsweredQuestion]


class RAGEvaluate:
    """Core RAG class used to evaluate RAG results."""
    def __init__(
        self,
        student_answer_path: str,
        dataset_path: str
    ) -> None:
        """
        Load data from file as Pydantic models.

        Args:
            student_answer_path (str): Path to data to evaluate
            dataset_path (str): Path to data to be compared to

        Returns:
            None: None
        """
        self.student = IOUtils.load_json_as_model(
            student_answer_path, StudentSearchResults
        )
        self.dataset = IOUtils.load_json_as_model(
            dataset_path, RagDataset
        )

    def validate_student(
        self,
        k: int,
        max_context_length: int
    ) -> bool:
        """
        Validate student dataset based on k and max_context_length.

        Args:
            k (int): The number of sources to be provided for each question.
            max_context_length (int): Maximum character size for each source.

        Returns:
            bool: Whether the whole student dataset passes all validation.
        """
        if not self.student.k == k:
            return False

        for entry in self.student.search_results:
            if len(entry.retrieved_sources) != k:
                return False
            for source in entry.retrieved_sources:
                if not (source.last_character_index
                        - source.first_character_index <= max_context_length):
                    return False

        return True

    def validate(
        self,
        k: int,
        max_context_length: int
    ) -> None:
        """
        Print info to terminal about evaluated data.

        Args:
            k (int): The number of sources to be provided for each question.
            max_context_length (int): Maximum character size for each source.

        Returns:
            None: Terminal output.
        """
        valid_len = self.validate_student(
            k, max_context_length
        )
        c1 = len(
            {entry.question_id for entry in self.dataset.rag_questions}
        )
        c2 = len(
            {entry.question_id for entry in self.dataset.rag_questions
             if getattr(entry, "sources")}
        )
        c3 = len(
            {entry.question_id for entry in self.student.search_results
             if getattr(entry, "retrieved_sources")}
        )

        print("Student data is valid:", valid_len)
        print("Total number of questions:", c1)
        print("Total number of questions with sources:", c2)
        print("Total number of questions with student sources:", c3)

    def count_evaluated(
        self
    ) -> int:
        """
        Return how many student questions were found in
        the evaluation dataset and can be evaluated.
        """
        count = 0

        for entry in self.student.search_results:
            gtruth = next((q for q in self.dataset.rag_questions
                          if q.question_id == entry.question_id), None)
            if gtruth and hasattr(gtruth, "sources"):
                count += 1

        return count

    def is_source_found(
        self,
        source: MinimalSource,
        truth_source: MinimalSource
    ) -> bool:
        """
        Check if student source matches evaluation dataset source,
        meaning they overlap at least RECALL_THRESHOLD (default: 5%).

        Args:
            source (MinimalSource): the student source to be evaluated.
            truth_source (MinimalSource): dataset source to compare.

        Returns:
            bool: Whether both sources overlap at least RECALL_THRESHOLD.
        """
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
        return ratio >= RECALL_THRESHOLD

    def recallat(
        self,
        k: int
    ) -> float:
        """
        Compute what proportion of the evaluation dataset's sources
        can be found in the student's top k retrieved sources.

        Args:
            k (int): The number of sources to check

        Returns:
            float: Proportion of found evaluation dataset sources
        """
        if not self.dataset.rag_questions:
            return 0.0

        scores = []
        gtruth_count = 0

        for i, entry in enumerate(self.student.search_results, 1):
            gtruth = next((q for q in self.dataset.rag_questions
                          if q.question_id == entry.question_id), None)

            if not gtruth or not hasattr(gtruth, "sources"):
                continue

            gtruth_count += 1

            if not entry.retrieved_sources:
                scores.append(0.0)
                continue

            unique_gtruth_found = set()
            for t_idx, gtruth_src in enumerate(gtruth.sources):
                for student_src in entry.retrieved_sources[:k]:
                    if self.is_source_found(student_src, gtruth_src):
                        print(f"src {i} found")
                        unique_gtruth_found.add(t_idx)
                        break

            scores.append(len(unique_gtruth_found) / len(gtruth.sources))

        return sum(scores) / gtruth_count if gtruth_count > 0 else 0.0

    def evaluate(
        self,
    ) -> None:
        """Compute recall@k values and print them to the terminal."""
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
