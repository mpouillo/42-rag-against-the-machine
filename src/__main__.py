#!/usr/bin/env python3

import fire
import sys

from .RAGInterface import RAGInterface


def main() -> None:
    fire.Fire(RAGInterface)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        sys.exit(f"Error: {e}")
