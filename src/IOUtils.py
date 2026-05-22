from pathlib import Path
from typing import List, Type, TypeVar
from pydantic import BaseModel
from collections import defaultdict

from .models import MinimalSource

T = TypeVar("T", bound=BaseModel)


class IOUtils:
    @staticmethod
    def load_json_as_model(
        file_path: str,
        model_cls: Type[T]
    ) -> T:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find file at {file_path}")
        return model_cls.model_validate_json(path.read_text())

    @staticmethod
    def save_object_as_json(
        file_path: str,
        obj: Type[T]
    ) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(obj.model_dump_json(indent=4))

    @staticmethod
    def get_text_from_file(
        file_path: str,
        first_character_index: int = 0,
        last_character_index: int | None = None
    ) -> str:
        text = Path(file_path).read_text()
        return text[first_character_index:last_character_index]

    @staticmethod
    def deduplicate_sources(
        sources: List[MinimalSource]
    ) -> List[MinimalSource]:

        file_groups = defaultdict(list)
        for src in sources:
            file_groups[src.file_path].append(src)

        retained_sources = []

        for file_path, group in file_groups.items():
            sorted_group = sorted(
                group,
                key=lambda x: (x.first_character_index - x.last_character_index)
            )

            files_retained = []

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
