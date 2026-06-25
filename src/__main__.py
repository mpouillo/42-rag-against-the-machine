#!/usr/bin/env python3

"""Entry point script for the RAG CLI application.

Initializes the Fire command line interface to expose RAGInterface commands.
"""

import fire
import sys

from .RAGInterface import RAGInterface


def main() -> None:
    """Execute the main application via Google Fire CLI."""
    fire.Fire(RAGInterface)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        sys.exit(f"Error: {e}")
