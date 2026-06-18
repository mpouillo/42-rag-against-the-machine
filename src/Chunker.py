import ast
import re

from typing import List

from .models import CodeChunk, MinimalSource


def chunk_python_file(
    file_path: str,
    source_code: str,
    max_chunk_size: int = 2000
) -> List[CodeChunk]:
    chunks = []

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return fallback_line_chunker(file_path, source_code, max_chunk_size)

    lines = source_code.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def get_char_indices(node: ast.AST) -> tuple[int, int]:
        start_line = getattr(node, "lineno", 1) - 1
        end_line = getattr(node, "end_lineno", start_line + 1) - 1

        start_char = line_offsets[start_line] + getattr(node, "col_offset", 0)
        end_char = line_offsets[end_line] + (
            getattr(node, "end_col_offset", len(lines[end_line]))
        )
        return start_char, end_char

    for node in tree.body:
        if isinstance(node, (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef
        )):
            start_idx, end_idx = get_char_indices(node)
            entity_text = source_code[start_idx:end_idx]

            lineage = f"File: {file_path}\n"
            if isinstance(node, ast.ClassDef):
                lineage += f"Class: {node.name}\n"
            else:
                lineage += f"Function: {node.name}\n"

            full_indexed_text = lineage + "\n" + entity_text

            if len(full_indexed_text) <= max_chunk_size:
                chunks.append(CodeChunk(
                    text=full_indexed_text,
                    source=MinimalSource(
                        file_path=file_path,
                        first_character_index=start_idx,
                        last_character_index=end_idx
                    )
                ))
            else:
                overlap = 400
                step = max_chunk_size - overlap - len(lineage)

                for i in range(0, len(entity_text), step):
                    sub_text = entity_text[
                        i:i + max_chunk_size - len(lineage)
                    ]
                    chunk_start = start_idx + i
                    chunk_end = min(chunk_start + len(sub_text), end_idx)

                    chunks.append(CodeChunk(
                        text=lineage + "\n" + sub_text,
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
    chunks = []
    heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    matches = list(heading_pattern.finditer(text))
    if not matches:
        return fallback_line_chunker(file_path, text, max_chunk_size)

    for idx, match in enumerate(matches):
        start_idx = match.start()
        end_idx = (
            matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        )

        section_text = text[start_idx:end_idx]
        heading_title = match.group(2)

        metadata_header = f"File: {file_path}\nSection: {heading_title}\n\n"
        full_indexed_text = metadata_header + section_text

        if len(full_indexed_text) <= max_chunk_size:
            chunks.append(CodeChunk(
                text=full_indexed_text,
                source=MinimalSource(
                    file_path=file_path,
                    first_character_index=start_idx,
                    last_character_index=end_idx
                )
            ))
        else:
            overlap = 300
            step = max_chunk_size - overlap - len(metadata_header)
            for i in range(0, len(section_text), step):
                sub_text = section_text[
                    i:i + max_chunk_size - len(metadata_header)
                ]
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
    chunks = []
    overlap = 300
    start = 0
    while start < len(text):
        end = min(start + max_chunk_size, len(text))
        chunks.append(CodeChunk(
            text=f"File: {file_path}\n\n" + text[start:end],
            source=MinimalSource(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end)
        ))
        if end == len(text):
            break
        start += (max_chunk_size - overlap)
    return chunks
