from collections import defaultdict
from pathlib import Path
from pydantic import BaseModel, ValidationError
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

        Args:
        - file_path (str): path to the file to be read
        - model_cls (Type[BaseModel]): Pydantic class with which
        to validate read data

        Returns:
            BaseModel: Pydantic object loaded from file data
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
        """
        Save Pydantic object to file.

        Args:
        - file_path (str): path where data should be saved
        - obj (BaseModel): Pydantic object to save to file

        Returns:
            None: Object data written to file
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
        """
        Return text data read from file.

        Args:
        - file_path (str): path to the file to be read
        - first_character_index (int) = 0: start index to return
        - last_character_index (int | None) = None: end index to return

        Returns:
            str: Text data read from file
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

    @staticmethod
    def deduplicate_sources(
        sources: List[MinimalSource],
    ) -> List[MinimalSource]:
        """
        Remove duplicate MinimalSources and merge overlapping/adjacent segments
        within the same file, without exceeding max_chunk_size.

        Original order is not guaranteed.

        Args:
            sources (List[MinimalSource]): List of MinimalSources to process

        Returns:
            List[MinimalSource]: Sorted, deduplicated, and merged sources
        """
        if not sources:
            return []

        max_chunk_size = max(
            src.last_character_index - src.first_character_index
            for src in sources
        )

        file_groups = defaultdict(list)
        for src in sources:
            file_groups[src.file_path].append(src)

        final_merged = []

        for file_path, file_srcs in file_groups.items():
            file_srcs.sort(key=lambda x: (
                x.first_character_index, -x.last_character_index
            ))

            current_merged = file_srcs[0]

            for next_src in file_srcs[1:]:
                curr_start = current_merged.first_character_index
                curr_end = current_merged.last_character_index

                next_start = next_src.first_character_index
                next_end = next_src.last_character_index

                if next_start >= curr_start and next_end <= curr_end:
                    continue

                elif next_start <= curr_end:
                    potential_end = max(curr_end, next_end)
                    potential_size = potential_end - curr_start

                    if potential_size <= max_chunk_size:
                        current_merged.last_character_index = potential_end
                    else:
                        final_merged.append(current_merged)
                        current_merged = next_src

                else:
                    final_merged.append(current_merged)
                    current_merged = next_src

            final_merged.append(current_merged)

        return final_merged

    @staticmethod
    def deduplicate_sources_and_keep_order(
        sources: List[MinimalSource],
    ) -> List[MinimalSource]:
        """
        Remove duplicate MinimalSources and merge overlapping/adjacent segments
        within the same file, without exceeding max_chunk_size.

        This version actually preserves the original order, except it gets me
        worse recall for some reason?? So whatever, it's here if needed
        during evaluation.

        Args:
            sources (List[MinimalSource]): List of MinimalSources to process

        Returns:
            List[MinimalSource]: Sorted, deduplicated, and merged sources
        """
        if not sources:
            return []

        max_chunk_size = max(
            src.last_character_index - src.first_character_index
            for src in sources
        )

        indexed_sources = list(enumerate(sources))

        file_groups = defaultdict(list)
        for original_idx, src in indexed_sources:
            file_groups[src.file_path].append((original_idx, src))

        final_sources = []

        for file_path, items in file_groups.items():
            items.sort(key=lambda x: (
                x[1].first_character_index, -x[1].last_character_index
            ))

            curr_orig_idx, current_src = items[0]
            current_merged = current_src.model_copy()

            for next_orig_idx, next_src in items[1:]:
                curr_start = current_merged.first_character_index
                curr_end = current_merged.last_character_index

                next_start = next_src.first_character_index
                next_end = next_src.last_character_index

                # Contained
                if next_start >= curr_start and next_end <= curr_end:
                    continue

                # Overlap
                elif next_start <= curr_end:
                    potential_end = max(curr_end, next_end)
                    if (potential_end - curr_start) <= max_chunk_size:
                        current_merged.last_character_index = potential_end
                        curr_orig_idx = min(curr_orig_idx, next_orig_idx)
                    else:
                        final_sources.append((curr_orig_idx, current_merged))
                        curr_orig_idx = next_orig_idx
                        current_merged = next_src.model_copy()

                # No overlap
                else:
                    final_sources.append((curr_orig_idx, current_merged))
                    curr_orig_idx = next_orig_idx
                    current_merged = next_src.model_copy()

            final_sources.append((curr_orig_idx, current_merged))

        final_sources.sort(key=lambda x: x[0])

        return [src for _, src in final_sources]
