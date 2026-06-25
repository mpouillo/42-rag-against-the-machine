"""Utilities for segmenting source code and markdown into manageable chunks."""

import ast
import re
from typing import List, Tuple

from .models import CodeChunk, MinimalSource


def chunk_python_file(
    file_path: str,
    source_code: str,
    max_chunk_size: int = 2000
) -> List[CodeChunk]:
    """Parse and chunk a Python source file using its AST structure.

    Args:
        file_path (str): The path of the target source file.
        source_code (str): Raw text string content of the Python script.
        max_chunk_size (int): Upper limit for character counts within a chunk.

    Returns:
        List[CodeChunk]: Text objects broken down by classes or functions.
    """
    chunks: List[CodeChunk] = []

    try:
        tree = ast.parse(source_code)
    except (SyntaxError, ValueError):
        return fallback_line_chunker(file_path, source_code, max_chunk_size)

    lines = source_code.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def get_char_indices(node: ast.AST) -> Tuple[int, int]:
        start_line = getattr(node, "lineno", 1) - 1
        end_line = getattr(node, "end_lineno", start_line + 1) - 1

        start_line = max(0, min(start_line, len(line_offsets) - 1))
        end_line = max(0, min(end_line, len(line_offsets) - 1))

        start_char = line_offsets[start_line] + getattr(
            node, "col_offset", 0
        )

        fallback_len = (
            len(lines[end_line]) if 0 <= end_line < len(lines) else 0
        )
        end_char = line_offsets[end_line] + getattr(
            node, "end_col_offset", fallback_len
        )
        return start_char, end_char

    for node in tree.body:
        if isinstance(node, (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
        )):
            start_idx, end_idx = get_char_indices(node)
            entity_text = source_code[start_idx:end_idx]

            metadata_header = f"File: {file_path}\n"
            if isinstance(node, ast.ClassDef):
                metadata_header += f"Class: {node.name}\n"
            else:
                metadata_header += f"Function: {node.name}\n"

            if (end_idx - start_idx) <= max_chunk_size:
                chunks.append(CodeChunk(
                    text=metadata_header + "\n" + entity_text,
                    source=MinimalSource(
                        file_path=file_path,
                        first_character_index=start_idx,
                        last_character_index=end_idx
                    )
                ))
            else:
                overlap = 400
                step = max_chunk_size - overlap
                if step <= 0:
                    step = max(1, max_chunk_size)

                for i in range(0, len(entity_text), step):
                    sub_text = entity_text[i:i + max_chunk_size]
                    chunk_start = start_idx + i
                    chunk_end = min(chunk_start + len(sub_text), end_idx)

                    chunks.append(CodeChunk(
                        text=metadata_header + "\n" + sub_text,
                        source=MinimalSource(
                            file_path=file_path,
                            first_character_index=chunk_start,
                            last_character_index=chunk_end
                        )
                    ))
                    if chunk_end == end_idx:
                        break
    return chunks


def chunk_markdown_file(
    file_path: str,
    text: str,
    max_chunk_size: int = 2000
) -> List[CodeChunk]:
    """Segment a Markdown documentation file based on heading landmarks.

    Args:
        file_path (str): The file path location string.
        text (str): Raw string content of the Markdown document.
        max_chunk_size (int): Maximum safe length of individual split segments.

    Returns:
        List[CodeChunk]: Chunks mapped accurately to Markdown sections.
    """
    chunks: List[CodeChunk] = []
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    matches = list(heading_pattern.finditer(text))
    if not matches:
        return fallback_line_chunker(file_path, text, max_chunk_size)

    for idx, match in enumerate(matches):
        start_idx = match.start()
        end_idx = (
            matches[idx + 1].start()
            if idx + 1 < len(matches)
            else len(text)
        )

        section_text = text[start_idx:end_idx]
        heading_title = match.group(2)

        metadata_header = f"File: {file_path}\nSection: {heading_title}\n\n"

        if (end_idx - start_idx) <= max_chunk_size:
            chunks.append(CodeChunk(
                text=metadata_header + section_text,
                source=MinimalSource(
                    file_path=file_path,
                    first_character_index=start_idx,
                    last_character_index=end_idx
                )
            ))
        else:
            overlap = 300
            step = max_chunk_size - overlap
            if step <= 0:
                step = max(1, max_chunk_size)

            for i in range(0, len(section_text), step):
                sub_text = section_text[i:i + max_chunk_size]
                chunk_start = start_idx + i
                chunk_end = min(chunk_start + len(sub_text), end_idx)

                chunks.append(CodeChunk(
                    text=metadata_header + sub_text,
                    source=MinimalSource(
                        file_path=file_path,
                        first_character_index=chunk_start,
                        last_character_index=chunk_end
                    )
                ))
                if chunk_end == end_idx:
                    break
    return chunks


def fallback_line_chunker(
    file_path: str,
    text: str,
    max_chunk_size: int
) -> List[CodeChunk]:
    """Provide a sliding-window chunking method when standard parsing fails.

    Args:
        file_path (str): Original relative file directory route.
        text (str): Raw string asset body sequence block.
        max_chunk_size (int): Absolute size limit threshold criteria.

    Returns:
        List[CodeChunk]: Sequential text components mapped by document steps.
    """
    chunks: List[CodeChunk] = []
    overlap = 300
    start = 0

    metadata_header = f"File: {file_path}\n\n"

    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(CodeChunk(
            text=metadata_header + text[start:end],
            source=MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end
            )
        ))
        if end == len(text):
            break

        step = max_chunk_size - overlap
        start += step if step > 0 else 1

    return chunks
