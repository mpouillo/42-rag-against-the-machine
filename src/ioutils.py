from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel

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
        start: int = 0,
        end: int | None = None
    ) -> str:
        return Path(file_path).read_text()[start:end]
