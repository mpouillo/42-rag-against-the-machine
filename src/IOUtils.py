"""Utility functions for file input/output and model serialization."""

from pathlib import Path
from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar

T = TypeVar("T", bound=BaseModel)


class IOUtils:
    """Helper class providing utility functions for IO/object management."""

    @staticmethod
    def load_json_as_model(
        file_path: str,
        model_cls: Type[T]
    ) -> T:
        """Read data from a JSON file and return a validated Pydantic object.

        Args:
            file_path (str): Path to the JSON file to be read.
            model_cls (Type[BaseModel]): Pydantic class used to
                validate the data.

        Returns:
            BaseModel: A populated Pydantic object loaded from the file data.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If the file data fails Pydantic validation
                or JSON parsing.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Could not find file at {path.resolve()}")

        try:
            file_content = path.read_text(encoding="utf-8")
            return model_cls.model_validate_json(file_content)
        except ValidationError as e:
            raise ValueError(f"Validation failed for {file_path}:\n{e}") from e
        except Exception as e:
            raise ValueError(
                f"Failed to read or parse data from {file_path}: {e}"
            ) from e

    @staticmethod
    def save_object_as_json(
        file_path: str,
        obj: BaseModel
    ) -> None:
        """Serialize and save a Pydantic object to a file as JSON.

        Args:
            file_path (str): Destination path where the data should be saved.
            obj (BaseModel): The Pydantic object to be saved to the file.

        Raises:
            OSError: If writing the file to the disk fails.
        """
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(obj.model_dump_json(indent=4), encoding="utf-8")
        except OSError as e:
            raise OSError(
                f"Failed to write Pydantic object data to {file_path}"
                f"Reason: {e}"
            ) from e

    @staticmethod
    def get_text_from_file(
        file_path: str,
        first_character_index: int = 0,
        last_character_index: int | None = None
    ) -> str:
        """Extract a specific sequence of characters from a text file.

        Args:
            file_path (str): Path to the target file.
            first_character_index (int): Starting index slice.
            last_character_index (int | None): Ending index slice.

        Returns:
            str: The text data read and sliced from the file.

        Raises:
            TypeError: If the provided indices are not integers or None.
            OSError: If the file cannot be read from the disk.
        """
        try:
            text = Path(file_path).read_text(encoding="utf-8")
            return text[first_character_index:last_character_index]
        except TypeError as e:
            raise TypeError(
                "Indices must be integers or None. "
                f"Got: {type(first_character_index)} "
                f"and {type(last_character_index)}"
            ) from e
        except OSError as e:
            raise OSError(
                f"Failed to read data from {file_path}. Reason: {e}"
            ) from e
