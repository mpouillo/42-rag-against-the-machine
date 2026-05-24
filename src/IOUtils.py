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
        sources: List[MinimalSource],
        max_chunk_size: int = 2000
    ) -> List[MinimalSource]:
        """
        Remove duplicate MinimalSources and merge overlapping/adjacent segments
        within the same file, without exceeding max_chunk_size.

        Args:
            sources (List[MinimalSource]): List of MinimalSources to process
            max_chunk_size (int): Maximum allowed character window size for merged items

        Returns:
            List[MinimalSource]: Sorted, deduplicated, and merged sources
        """
        if not sources:
            return []

        # 1. Group sources purely by their target file path
        grouped_by_file = defaultdict(list)
        for src in sources:
            grouped_by_file[src.file_path].append(src)

        final_merged_sources = []

        # 2. Process each file's intervals independently
        for file_path, file_srcs in grouped_by_file.items():
            # Sort intervals strictly by their start index.
            # If starts match, sort by descending end index to process larger spans first.
            file_srcs.sort(key=lambda x: (x.first_character_index, -x.last_character_index))

            # Initialize our rolling merge window with the first source segment
            current_merged = file_srcs[0]

            for next_src in file_srcs[1:]:
                # Quick aliases for clean reading
                curr_start = current_merged.first_character_index
                curr_end = current_merged.last_character_index

                next_start = next_src.first_character_index
                next_end = next_src.last_character_index

                # Scenario A: The next source is fully swallowed/contained (Duplicate or Nested)
                if next_start >= curr_start and next_end <= curr_end:
                    continue

                # Scenario B: Overlapping or perfectly adjacent intervals
                elif next_start <= curr_end:
                    # Calculate what the potential new size would be if we combined them
                    potential_end = max(curr_end, next_end)
                    potential_size = potential_end - curr_start

                    if potential_size <= max_chunk_size:
                        # Within limit -> Extend the current active window boundary
                        current_merged.last_character_index = potential_end
                    else:
                        # Size limit broken -> Close active window, start a new one
                        final_merged_sources.append(current_merged)
                        current_merged = next_src

                # Scenario C: Completely disjoint intervals (a gap exists between them)
                else:
                    final_merged_sources.append(current_merged)
                    current_merged = next_src

            # Don't forget to save the final open interval after the loop exits
            final_merged_sources.append(current_merged)

        return final_merged_sources
