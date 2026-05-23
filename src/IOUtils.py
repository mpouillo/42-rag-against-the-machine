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
        Deduplicate MinimalSources by removing those encompassed by other ones.

        Returns a list of deduplicated MinimalSources.

        Keyword arguments:
        - sources (List[MinimalSource]): list of MinimalSources to filter
        """

        file_groups = defaultdict(list)
        for src in sources:
            file_groups[src.file_path].append(src)

        retained_sources: List[MinimalSource] = []

        for file_path, group in file_groups.items():
            sorted_group = sorted(
                group,
                key=lambda x: (
                    x.first_character_index - x.last_character_index
                )
            )

            files_retained: List[MinimalSource] = []

            for current in sorted_group:
                is_contained = False

                for kept in files_retained:
                    if (
                        current.first_character_index
                        >= kept.first_character_index
                        and current.last_character_index
                        <= kept.last_character_index
                    ):
                        is_contained = True
                        break

                if not is_contained:
                    files_retained.append(current)

            retained_sources += files_retained

        return retained_sources
