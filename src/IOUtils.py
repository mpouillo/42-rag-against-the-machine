from collections import defaultdict
from pathlib import Path
from pydantic import BaseModel
from typing import List, Type, TypeVar

from .models import MinimalSource

T = TypeVar("T", bound=BaseModel)


class IOUtils:
    """Helper class providing utiliy functions for IO and object management."""
    @staticmethod
    def load_json_as_model(
        file_path: str,
        model_cls: Type[T]
    ) -> T:
        """
        Read data from file and return a Pydantic object.

        Keyword arguments:
        - file_path (str): path to the file to be read
        - model_cls (Type[BaseModel]): Pydantic class with which
        to validate read data
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find file at {file_path}")
        return model_cls.model_validate_json(path.read_text())

    @staticmethod
    def save_object_as_json(
        file_path: str,
        obj: T
    ) -> None:
        """
        Save Pydantic object to file.

        Keyword arguments:
        - file_path (str): path where data should be saved
        - obj (BaseModel): Pydantic object to save to file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(obj.model_dump_json(indent=4))

    @staticmethod
    def get_text_from_file(
        file_path: str,
        first_character_index: int = 0,
        last_character_index: int | None = None
    ) -> str:
        """
        Return text data read from file.

        Keyword arguments:
        - file_path (str): path to the file to be read
        - first_character_index (int) = 0: start index to return
        - last_character_index (int | None) = None: end index to return
        """
        text = Path(file_path).read_text()
        return text[first_character_index:last_character_index]

    @staticmethod
    def deduplicate_sources(
        sources: List[MinimalSource]
    ) -> List[MinimalSource]:
        """
        Remove duplicate MinimalSources contained within other ones.

        Args:
            sources (List[MinimalSource]): list of MinimalSources to filter

        Returns:
            List[MinimalSource]: Deduplicated sources
        """
        if not sources:
            return []

        grouped_by_file = defaultdict(list)
        for idx, src in enumerate(sources):
            grouped_by_file[src.file_path].append((idx, src))

        swallowed_indices = set()

        for file_path, items in grouped_by_file.items():
            if len(items) < 2:
                continue

            for i in range(len(items)):
                idx_a, src_a = items[i]
                start_a = src_a.first_character_index
                end_a = src_a.last_character_index

                for j in range(len(items)):
                    if i == j:
                        continue

                    idx_b, src_b = items[j]
                    start_b = src_b.first_character_index
                    end_b = src_b.last_character_index

                    if start_b <= start_a and end_b >= end_a:
                        if (
                            start_b == start_a
                            and end_b == end_a
                            and idx_b > idx_a
                        ):
                            continue

                        swallowed_indices.add(idx_a)
                        break

        return [src for idx, src in enumerate(sources)
                if idx not in swallowed_indices]
